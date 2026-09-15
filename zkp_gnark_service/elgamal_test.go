package main

import (
	"math/big"
	"testing"

	edbn254 "github.com/consensys/gnark-crypto/ecc/bn254/twistededwards"
)

func mustKeygen(t *testing.T) *big.Int {
	t.Helper()
	sk, _, err := elgamalKeygen()
	if err != nil {
		t.Fatal(err)
	}
	return sk
}

// plainTestGlobal is a public initial model with the given quantized values.
func plainTestGlobal(t *testing.T, qs []int64) (elgamalGlobal, []int64) {
	t.Helper()
	glob, err := plainGlobal(qs)
	if err != nil {
		t.Fatal(err)
	}
	return glob, qs
}

func zeroGlobal(t *testing.T, n int) (elgamalGlobal, []int64) {
	return plainTestGlobal(t, make([]int64, n))
}

func TestElgamalHonestRoundTripAcrossTwoRounds(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	ctx := big.NewInt(42)

	// Round 1: updates measured against the public initial model.
	init := []int64{10, -20, 30}
	glob, sums := plainTestGlobal(t, init)
	a := []int64{13, -27, 80}
	b := []int64{9, -18, -30}
	bound := big.NewInt(10_000)
	ctsA, proofA, err := elgamalProve(pk, sk, a, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	ctsB, proofB, err := elgamalProve(pk, sk, b, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]struct {
		cts   []elgamalCiphertext
		proof []byte
	}{"a": {ctsA, proofA}, "b": {ctsB, proofB}} {
		ok, err := elgamalVerify(pk, c.cts, glob, bound, ctx, c.proof)
		if err != nil || !ok {
			t.Fatalf("honest proof %s did not verify: %v", name, err)
		}
	}

	agg, err := elgamalAggregate([][]elgamalCiphertext{ctsA, ctsB}, []int64{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	offset := new(big.Int).Mul(big.NewInt(5), elgamalOffset)
	got, err := elgamalDecrypt(sk, agg, offset, 5*elgamalOffset.Int64())
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if want := 2*a[i] + 3*b[i]; got[i] != want {
			t.Fatalf("coordinate %d: got %d want %d", i, got[i], want)
		}
	}

	// Round 2: the global model is the encrypted aggregate, W = 5. The client
	// knows the sums from decryption; the verifier only has the ciphertexts.
	next := elgamalGlobal{Cts: agg, Weight: 5}
	c := []int64{(got[0] + 2) / 5, got[1] / 5, got[2] / 5}
	energy := int64(0)
	for i := range c {
		d := 5*c[i] - got[i]
		energy += d * d
	}
	ctsC, proofC, err := elgamalProve(pk, sk, c, next, got, big.NewInt(energy), ctx)
	if err != nil {
		t.Fatalf("round-2 update at its exact energy was refused: %v", err)
	}
	if ok, err := elgamalVerify(pk, ctsC, next, big.NewInt(energy), ctx, proofC); err != nil || !ok {
		t.Fatalf("round-2 proof against the aggregate did not verify: %v", err)
	}
	if _, _, err := elgamalProve(pk, sk, c, next, got, big.NewInt(energy-1), ctx); err == nil && energy > 0 {
		t.Fatal("round-2 update above the bound was proved")
	}
}

func TestElgamalBoundIsOnTheUpdateNotTheWeights(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	far := []int64{100_000, -100_000}
	glob, sums := plainTestGlobal(t, far)
	// Large weights, zero update: fits a bound of 1.
	if _, _, err := elgamalProve(pk, sk, far, glob, sums, big.NewInt(1), big.NewInt(0)); err != nil {
		t.Fatalf("zero update far from the origin was refused: %v", err)
	}
	// Small weights, large update: refused.
	if _, _, err := elgamalProve(pk, sk, []int64{0, 0}, glob, sums, big.NewInt(1_000_000), big.NewInt(0)); err == nil {
		t.Fatal("update of norm ~141k was proved under a bound of 1000")
	}
}

func TestElgamalProofIsBoundToTheGlobalModel(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(1_000_000), big.NewInt(1)
	glob, sums := plainTestGlobal(t, []int64{5, 5})
	cts, proof, err := elgamalProve(pk, sk, []int64{6, 4}, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	other, _ := plainTestGlobal(t, []int64{5, 6})
	if ok, _ := elgamalVerify(pk, cts, other, bound, ctx, proof); ok {
		t.Fatal("proof verified against a different global model")
	}
	aggCts, _, err := elgamalEncrypt(pk, []int64{5, 5})
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, cts, elgamalGlobal{Cts: aggCts, Weight: 1}, bound, ctx, proof); ok {
		t.Fatal("proof verified against a re-randomised encryption of the same global model")
	}
	if ok, _ := elgamalVerify(pk, cts, elgamalGlobal{Cts: glob.Cts, Weight: 2}, bound, ctx, proof); ok {
		t.Fatal("proof verified under a different global weight")
	}
}

func TestElgamalProverRefusesSumsThatDoNotDecryptTheGlobal(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	agg, _, err := elgamalEncrypt(pk, []int64{7, -7})
	if err != nil {
		t.Fatal(err)
	}
	glob := elgamalGlobal{Cts: agg, Weight: 1}
	if _, _, err := elgamalProve(pk, sk, []int64{7, -7}, glob, []int64{7, -7}, big.NewInt(1), big.NewInt(0)); err != nil {
		t.Fatalf("correct sums refused: %v", err)
	}
	if _, _, err := elgamalProve(pk, sk, []int64{0, 0}, glob, []int64{0, 0}, big.NewInt(1), big.NewInt(0)); err == nil {
		t.Fatal("claimed global sums that don't decrypt the aggregate were accepted")
	}
	if _, _, err := elgamalProve(pk, mustKeygen(t), []int64{7, -7}, glob, []int64{7, -7}, big.NewInt(1), big.NewInt(0)); err == nil {
		t.Fatal("a secret key that doesn't match the public key was accepted")
	}
}

func TestElgamalRejectsCiphertextOfAnotherVector(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(10_000), big.NewInt(1)
	glob, sums := zeroGlobal(t, 2)

	_, proofA, err := elgamalProve(pk, sk, []int64{1, 2}, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	ctsB, _, err := elgamalProve(pk, sk, []int64{1, 3}, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, ctsB, glob, bound, ctx, proofA); ok {
		t.Fatal("proof over A verified against a ciphertext of B")
	}
}

func TestElgamalProofIsBoundToPublicInputs(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(10_000), big.NewInt(7)
	glob, sums := zeroGlobal(t, 2)
	cts, proof, err := elgamalProve(pk, sk, []int64{4, -4}, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, cts, glob, bound, big.NewInt(8), proof); ok {
		t.Fatal("proof verified under a different context")
	}
	if ok, _ := elgamalVerify(pk, cts, glob, big.NewInt(1_000_000), ctx, proof); ok {
		t.Fatal("proof verified under a different bound")
	}
	otherPK := mulBase(mustKeygen(t))
	if ok, _ := elgamalVerify(otherPK, cts, glob, bound, ctx, proof); ok {
		t.Fatal("proof verified under a different public key")
	}
	swapped := []elgamalCiphertext{cts[1], cts[0]}
	if ok, _ := elgamalVerify(pk, swapped, glob, bound, ctx, proof); ok {
		t.Fatal("proof verified with reordered ciphertexts")
	}
}

func TestElgamalRefusesOverBoundNorm(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	glob, sums := zeroGlobal(t, 2)
	// 100² + 100² = 20000 > 19999
	if _, _, err := elgamalProve(pk, sk, []int64{100, 100}, glob, sums, big.NewInt(19_999), big.NewInt(0)); err == nil {
		t.Fatal("over-bound vector was proved")
	}
	if _, _, err := elgamalProve(pk, sk, []int64{100, 100}, glob, sums, big.NewInt(20_000), big.NewInt(0)); err != nil {
		t.Fatalf("vector at the bound was refused: %v", err)
	}
}

func TestElgamalRefusesOutOfRangeValues(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	glob, sums := zeroGlobal(t, 2)
	limit := elgamalOffset.Int64()
	huge := new(big.Int).Lsh(big.NewInt(1), 60)
	for _, q := range []int64{limit, -limit - 1} {
		if _, _, err := elgamalProve(pk, sk, []int64{q, 0}, glob, sums, huge, big.NewInt(0)); err == nil {
			t.Fatalf("out-of-range value %d was proved", q)
		}
	}
}

func TestElgamalAggregateRefusesWeightThatWouldOverflowTheCircuit(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	cts, _, err := elgamalEncrypt(pk, []int64{1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := elgamalAggregate([][]elgamalCiphertext{cts, cts}, []int64{1 << 13, 1 << 13}); err == nil {
		t.Fatal("total weight 2^14 was aggregated")
	}
	if _, err := elgamalAggregate([][]elgamalCiphertext{cts, cts}, []int64{1 << 13, (1 << 13) - 1}); err != nil {
		t.Fatalf("total weight 2^14 - 1 refused: %v", err)
	}
}

func TestDecodeRejectsNonCanonicalPoints(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	enc := pk.Bytes()
	if _, err := decodePublicKey(enc[:]); err != nil {
		t.Fatalf("canonical key rejected: %v", err)
	}
	bad := enc
	bad[0] ^= 0xff
	if _, err := decodePoint(bad[:]); err == nil {
		if p, _ := decodePoint(bad[:]); p.Equal(&pk) {
			t.Fatal("altered encoding decoded to the same point")
		}
	}
}

func TestElgamalCommitThenProveSampledCoordinates(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(1_000_000), big.NewInt(9)
	qs := []int64{5, -3, 700, 12}
	glob, sums := plainTestGlobal(t, []int64{4, -2, 690, 10})

	committed, rands, err := elgamalEncrypt(pk, qs)
	if err != nil {
		t.Fatal(err)
	}
	// Challenge picks coordinates 1 and 2; prove them with the stored randomness.
	sampled := []elgamalCiphertext{committed[1], committed[2]}
	sampledGlobal := elgamalGlobal{Cts: []elgamalCiphertext{glob.Cts[1], glob.Cts[2]}, Weight: 1}
	recomputed, proof, err := elgamalProveWith(pk, sk, []int64{qs[1], qs[2]}, []*big.Int{rands[1], rands[2]}, sampledGlobal, []int64{sums[1], sums[2]}, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := range sampled {
		if !recomputed[i].C1.Equal(&sampled[i].C1) || !recomputed[i].C2.Equal(&sampled[i].C2) {
			t.Fatalf("prove_with ciphertext %d differs from the commitment", i)
		}
	}
	if ok, err := elgamalVerify(pk, sampled, sampledGlobal, bound, ctx, proof); err != nil || !ok {
		t.Fatalf("proof over committed ciphertexts did not verify: %v", err)
	}
}

func TestElgamalProofWithWrongRandomnessFailsAgainstCommitment(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(1_000_000), big.NewInt(1)
	qs := []int64{4, 8}
	glob, sums := zeroGlobal(t, 2)
	committed, _, err := elgamalEncrypt(pk, qs)
	if err != nil {
		t.Fatal(err)
	}
	_, freshRands, _ := elgamalEncrypt(pk, qs)
	_, proof, err := elgamalProveWith(pk, sk, qs, freshRands, glob, sums, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, committed, glob, bound, ctx, proof); ok {
		t.Fatal("proof with different randomness verified against the committed ciphertexts")
	}
}

func TestElgamalProveWithRejectsOutOfRangeRandomness(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	params := edParams()
	tooBig := new(big.Int).Set(&params.Order)
	glob, sums := zeroGlobal(t, 1)
	if _, _, err := elgamalProveWith(pk, sk, []int64{1}, []*big.Int{tooBig}, glob, sums, big.NewInt(10), big.NewInt(0)); err == nil {
		t.Fatal("randomness equal to the group order was accepted")
	}
}

var _ = edbn254.PointAffine{}
