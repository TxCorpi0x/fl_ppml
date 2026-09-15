package main

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
)

// proveRawWitness bypasses elgamalProve's range pre-check, the way a malicious
// client running its own prover would, so only the circuit can refuse.
func proveRawWitness(t *testing.T, vs []*big.Int) error {
	t.Helper()
	_, pk, err := elgamalKeygen()
	if err != nil {
		t.Fatal(err)
	}
	k, err := store.keys(elgamalCircuitID)
	if err != nil {
		t.Fatal(err)
	}
	if len(vs) > k.n {
		t.Fatalf("test witness longer than circuit size %d", k.n)
	}
	a := newElgamalCircuit(k.n)
	a.PKX, a.PKY = coord(pk.X), coord(pk.Y)
	a.Bound, a.Context = new(big.Int).Lsh(big.NewInt(1), 200), big.NewInt(0)
	params := edParams()
	pad := paddingCiphertext(pk)
	for i := 0; i < k.n; i++ {
		if i >= len(vs) {
			a.Values[i], a.Rand[i] = new(big.Int).Set(elgamalOffset), big.NewInt(0)
			a.C1X[i], a.C1Y[i] = coord(pad.C1.X), coord(pad.C1.Y)
			a.C2X[i], a.C2Y[i] = coord(pad.C2.X), coord(pad.C2.Y)
			continue
		}
		r, err := randomScalar()
		if err != nil {
			t.Fatal(err)
		}
		vMod := new(big.Int).Mod(vs[i], &params.Order) // consistent ciphertext for the out-of-range value
		vg := mulBase(vMod)
		rpk, c2 := pk, pk
		rpk.ScalarMultiplication(&pk, r)
		c2.Add(&vg, &rpk)
		c1 := mulBase(r)
		a.Values[i], a.Rand[i] = new(big.Int).Mod(vs[i], bn254R), r
		a.C1X[i], a.C1Y[i] = coord(c1.X), coord(c1.Y)
		a.C2X[i], a.C2Y[i] = coord(c2.X), coord(c2.Y)
	}
	w, err := frontend.NewWitness(a, ecc.BN254.ScalarField())
	if err != nil {
		return err
	}
	_, err = groth16.Prove(k.cs, k.pk, w)
	return err
}

func TestElgamalCircuitAcceptsInRangeRawWitness(t *testing.T) {
	top := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), elgamalValueBits), big.NewInt(1))
	if err := proveRawWitness(t, []*big.Int{big.NewInt(0), top}); err != nil {
		t.Fatalf("in-range raw witness refused: %v", err)
	}
}

func TestElgamalCircuitRejectsOutOfRangeRawWitness(t *testing.T) {
	cases := map[string]*big.Int{
		"v = 2^18": new(big.Int).Lsh(big.NewInt(1), elgamalValueBits),
		"v = -1":   big.NewInt(-1),
	}
	for name, v := range cases {
		if err := proveRawWitness(t, []*big.Int{v, big.NewInt(0)}); err == nil {
			t.Fatalf("%s: circuit accepted an out-of-range value", name)
		}
	}
}
