package main

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
)

// proveRawWitness bypasses elgamalProve's native pre-checks, the way a
// malicious client running its own prover would, so only the circuit can
// refuse. The global model is a public initial model of zeros (W = 1); mutate
// may then corrupt any witness field.
func proveRawWitness(t *testing.T, vs []*big.Int, mutate func(a *elgamalCircuit, sk *big.Int)) error {
	t.Helper()
	sk, pk, err := elgamalKeygen()
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
	a.Weight, a.SK = big.NewInt(1), sk
	params := edParams()
	pad, gpad := paddingCiphertext(pk), paddingGlobal(1)
	for i := 0; i < k.n; i++ {
		a.Agg[i] = new(big.Int).Set(elgamalOffset) // global plaintext q = 0
		if i >= len(vs) {
			a.Values[i], a.Rand[i] = new(big.Int).Set(elgamalOffset), big.NewInt(0)
			setSlot(a, i, pad, gpad)
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
		a.Values[i], a.Rand[i] = new(big.Int).Mod(vs[i], bn254R), r
		setSlot(a, i, elgamalCiphertext{mulBase(r), c2}, gpad)
	}
	if mutate != nil {
		mutate(a, sk)
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
	if err := proveRawWitness(t, []*big.Int{big.NewInt(0), top}, nil); err != nil {
		t.Fatalf("in-range raw witness refused: %v", err)
	}
}

func TestElgamalCircuitRejectsOutOfRangeRawWitness(t *testing.T) {
	cases := map[string]*big.Int{
		"v = 2^18": new(big.Int).Lsh(big.NewInt(1), elgamalValueBits),
		"v = -1":   big.NewInt(-1),
	}
	for name, v := range cases {
		if err := proveRawWitness(t, []*big.Int{v, big.NewInt(0)}, nil); err == nil {
			t.Fatalf("%s: circuit accepted an out-of-range value", name)
		}
	}
}

// A client that lies about the global model's plaintext, to make its update
// look small, must be refused by the circuit itself.
func TestElgamalCircuitRejectsFalseGlobalPlaintext(t *testing.T) {
	v := new(big.Int).Add(elgamalOffset, big.NewInt(5_000))
	cases := map[string]func(a *elgamalCircuit, sk *big.Int){
		"T that doesn't decrypt the global slot": func(a *elgamalCircuit, _ *big.Int) {
			a.Agg[0] = new(big.Int).Set(v) // claims the global is already at v, so the update is 0
		},
		"T aliased by the group order": func(a *elgamalCircuit, _ *big.Int) {
			params := edParams()
			a.Agg[0] = new(big.Int).Add(elgamalOffset, &params.Order)
		},
		"secret key that doesn't match PK": func(a *elgamalCircuit, sk *big.Int) {
			a.SK = new(big.Int).Add(sk, big.NewInt(1))
		},
		"weight of 2^14": func(a *elgamalCircuit, _ *big.Int) {
			a.Weight = big.NewInt(1 << elgamalWeightBits)
		},
	}
	for name, mutate := range cases {
		err := proveRawWitness(t, []*big.Int{v}, func(a *elgamalCircuit, sk *big.Int) {
			a.Bound = big.NewInt(1) // only a zero update fits
			mutate(a, sk)
		})
		if err == nil {
			t.Fatalf("%s: circuit accepted it", name)
		}
	}
}
