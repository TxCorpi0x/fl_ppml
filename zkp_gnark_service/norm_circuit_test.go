package main

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
)

// proveNormRaw proves the norm circuit for raw field elements, bypassing
// decodeWeights (which only produces int64 values), the way a client running
// its own prover could. The hash is computed over the same elements, so only
// the circuit's own constraints can refuse.
func proveNormRaw(t *testing.T, values []*big.Int, bound *big.Int) error {
	t.Helper()
	k, err := store.keys(normCircuitID)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) > k.n {
		t.Fatalf("test vector longer than circuit size %d", k.n)
	}
	padded := make([]*big.Int, k.n)
	for i := range padded {
		padded[i] = big.NewInt(0)
		if i < len(values) {
			padded[i] = new(big.Int).Mod(values[i], bn254R)
		}
	}
	hash, err := computeHash(padded)
	if err != nil {
		t.Fatal(err)
	}
	a := &proofCircuit{Weights: make([]frontend.Variable, k.n), Bound: bound, Hash: hash}
	for i, v := range padded {
		a.Weights[i] = v
	}
	w, err := frontend.NewWitness(a, ecc.BN254.ScalarField())
	if err != nil {
		return err
	}
	_, err = groth16.Prove(k.cs, k.pk, w)
	return err
}

func TestNormCircuitAcceptsInt64Extremes(t *testing.T) {
	top := new(big.Int).Sub(int64Offset, big.NewInt(1)) // 2^63 − 1
	bottom := new(big.Int).Neg(int64Offset)             // −2^63
	bound := new(big.Int).Lsh(big.NewInt(1), 128)
	if err := proveNormRaw(t, []*big.Int{top, bottom}, bound); err != nil {
		t.Fatalf("int64 extremes refused: %v", err)
	}
}

func TestNormCircuitRejectsValuesOutsideInt64(t *testing.T) {
	bound := new(big.Int).Lsh(big.NewInt(1), 200)
	for name, v := range map[string]*big.Int{
		"2^63":    new(big.Int).Set(int64Offset),
		"−2^63−1": new(big.Int).Sub(new(big.Int).Neg(int64Offset), big.NewInt(1)),
	} {
		if err := proveNormRaw(t, []*big.Int{v}, bound); err == nil {
			t.Fatalf("%s: circuit accepted a value outside int64", name)
		}
	}
}

// Two large field elements whose squares sum to 5 modulo the field: without a
// range check they satisfy Σw² ≤ 5.
func TestNormCircuitRejectsSquaresThatWrapTheField(t *testing.T) {
	var w1, w2, sq, target, small, one fr.Element
	small.SetUint64(5)
	one.SetOne()
	w1.SetBigInt(new(big.Int).Lsh(big.NewInt(1), 100))
	for {
		sq.Square(&w1)
		target.Sub(&small, &sq)
		if w2.Sqrt(&target) != nil {
			break
		}
		w1.Add(&w1, &one)
	}
	var check, sq2 fr.Element
	sq2.Square(&w2)
	check.Add(&sq, &sq2)
	if !check.Equal(&small) {
		t.Fatal("test construction: squares don't sum to 5 modulo the field")
	}
	values := []*big.Int{w1.BigInt(new(big.Int)), w2.BigInt(new(big.Int))}
	if err := proveNormRaw(t, values, big.NewInt(5)); err == nil {
		t.Fatal("circuit accepted field elements whose squares wrap to a small sum")
	}
}
