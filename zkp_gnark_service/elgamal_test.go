package main

import (
	"math/big"
	"testing"
)

func mustKeygen(t *testing.T) *big.Int {
	t.Helper()
	sk, _, err := elgamalKeygen()
	if err != nil {
		t.Fatal(err)
	}
	return sk
}

func TestElgamalHonestRoundTrip(t *testing.T) {
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(10_000), big.NewInt(42)

	a := []int64{3, -7, 50}
	b := []int64{-1, 2, -60}
	ctsA, proofA, err := elgamalProve(pk, a, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	ctsB, proofB, err := elgamalProve(pk, b, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	for name, c := range map[string]struct {
		cts   []elgamalCiphertext
		proof []byte
	}{"a": {ctsA, proofA}, "b": {ctsB, proofB}} {
		ok, err := elgamalVerify(pk, c.cts, bound, ctx, c.proof)
		if err != nil || !ok {
			t.Fatalf("honest proof %s did not verify: %v", name, err)
		}
	}

	sum, err := elgamalAggregate([][]elgamalCiphertext{ctsA, ctsB}, []int64{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	offset := new(big.Int).Mul(big.NewInt(5), elgamalOffset)
	got, err := elgamalDecrypt(sk, sum, offset, 5*elgamalOffset.Int64())
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if want := 2*a[i] + 3*b[i]; got[i] != want {
			t.Fatalf("coordinate %d: got %d want %d", i, got[i], want)
		}
	}
}

func TestElgamalRejectsCiphertextOfAnotherVector(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	bound, ctx := big.NewInt(10_000), big.NewInt(1)

	_, proofA, err := elgamalProve(pk, []int64{1, 2}, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	ctsB, _, err := elgamalProve(pk, []int64{1, 3}, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, ctsB, bound, ctx, proofA); ok {
		t.Fatal("proof over A verified against a ciphertext of B")
	}
}

func TestElgamalProofIsBoundToPublicInputs(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	bound, ctx := big.NewInt(10_000), big.NewInt(7)
	cts, proof, err := elgamalProve(pk, []int64{4, -4}, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, cts, bound, big.NewInt(8), proof); ok {
		t.Fatal("proof verified under a different context")
	}
	if ok, _ := elgamalVerify(pk, cts, big.NewInt(1_000_000), ctx, proof); ok {
		t.Fatal("proof verified under a different bound")
	}
	otherPK := mulBase(mustKeygen(t))
	if ok, _ := elgamalVerify(otherPK, cts, bound, ctx, proof); ok {
		t.Fatal("proof verified under a different public key")
	}
	swapped := []elgamalCiphertext{cts[1], cts[0]}
	if ok, _ := elgamalVerify(pk, swapped, bound, ctx, proof); ok {
		t.Fatal("proof verified with reordered ciphertexts")
	}
}

func TestElgamalRefusesOverBoundNorm(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	// 100² + 100² = 20000 > 19999
	if _, _, err := elgamalProve(pk, []int64{100, 100}, big.NewInt(19_999), big.NewInt(0)); err == nil {
		t.Fatal("over-bound vector was proved")
	}
	if _, _, err := elgamalProve(pk, []int64{100, 100}, big.NewInt(20_000), big.NewInt(0)); err != nil {
		t.Fatalf("vector at the bound was refused: %v", err)
	}
}

func TestElgamalRefusesOutOfRangeValues(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	limit := elgamalOffset.Int64()
	huge := new(big.Int).Lsh(big.NewInt(1), 60)
	for _, q := range []int64{limit, -limit - 1} {
		if _, _, err := elgamalProve(pk, []int64{q, 0}, huge, big.NewInt(0)); err == nil {
			t.Fatalf("out-of-range value %d was proved", q)
		}
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
	pk := mulBase(mustKeygen(t))
	bound, ctx := big.NewInt(1_000_000), big.NewInt(9)
	qs := []int64{5, -3, 700, 12}

	committed, rands, err := elgamalEncrypt(pk, qs)
	if err != nil {
		t.Fatal(err)
	}
	// Challenge picks coordinates 1 and 2; prove them with the stored randomness.
	sampled := []elgamalCiphertext{committed[1], committed[2]}
	recomputed, proof, err := elgamalProveWith(pk, []int64{qs[1], qs[2]}, []*big.Int{rands[1], rands[2]}, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i := range sampled {
		if !recomputed[i].C1.Equal(&sampled[i].C1) || !recomputed[i].C2.Equal(&sampled[i].C2) {
			t.Fatalf("prove_with ciphertext %d differs from the commitment", i)
		}
	}
	if ok, err := elgamalVerify(pk, sampled, bound, ctx, proof); err != nil || !ok {
		t.Fatalf("proof over committed ciphertexts did not verify: %v", err)
	}
}

func TestElgamalProofWithWrongRandomnessFailsAgainstCommitment(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	bound, ctx := big.NewInt(1_000_000), big.NewInt(1)
	qs := []int64{4, 8}
	committed, _, err := elgamalEncrypt(pk, qs)
	if err != nil {
		t.Fatal(err)
	}
	_, freshRands, _ := elgamalEncrypt(pk, qs)
	_, proof, err := elgamalProveWith(pk, qs, freshRands, bound, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := elgamalVerify(pk, committed, bound, ctx, proof); ok {
		t.Fatal("proof with different randomness verified against the committed ciphertexts")
	}
}

func TestElgamalProveWithRejectsOutOfRangeRandomness(t *testing.T) {
	pk := mulBase(mustKeygen(t))
	params := edParams()
	tooBig := new(big.Int).Set(&params.Order)
	if _, _, err := elgamalProveWith(pk, []int64{1}, []*big.Int{tooBig}, big.NewInt(10), big.NewInt(0)); err == nil {
		t.Fatal("randomness equal to the group order was accepted")
	}
}
