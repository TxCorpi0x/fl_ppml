package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/consensys/gnark/backend/groth16"
)

const testNormN, testElgamalN = 8, 4

// buildTestStore runs setup in memory with both key halves, so package tests
// can prove and verify in one process. Production never does this.
func buildTestStore(normN, elgamalN int) (*keyStore, error) {
	s := &keyStore{role: "test", manifestHash: "test", byCircuit: map[string]*circuitKeys{}}
	for id, n := range map[string]int{normCircuitID: normN, elgamalCircuitID: elgamalN} {
		cs, err := compileCircuit(id, n)
		if err != nil {
			return nil, err
		}
		pk, vk, err := groth16.Setup(cs)
		if err != nil {
			return nil, err
		}
		var buf bytes.Buffer
		if _, err := vk.WriteTo(&buf); err != nil {
			return nil, err
		}
		s.byCircuit[id] = &circuitKeys{n: n, cs: cs, pk: pk, vk: vk, vkHash: sha256Hex(buf.Bytes())}
	}
	return s, nil
}

func TestMain(m *testing.M) {
	s, err := buildTestStore(testNormN, testElgamalN)
	if err != nil {
		panic(err)
	}
	store = s
	os.Exit(m.Run())
}

func withStore(t *testing.T, s *keyStore) {
	t.Helper()
	previous := store
	store = s
	t.Cleanup(func() { store = previous })
}

func runTestSetup(t *testing.T) (keysDir, pkDir string) {
	t.Helper()
	dir := t.TempDir()
	keysDir, pkDir = filepath.Join(dir, "keys"), filepath.Join(dir, "pk")
	if err := runSetup([]string{"--keys-dir", keysDir, "--pk-dir", pkDir, "--norm-n", "4", "--elgamal-n", "2"}); err != nil {
		t.Fatal(err)
	}
	return keysDir, pkDir
}

func TestSetupThenLoadEachRoleWithOnlyItsKeys(t *testing.T) {
	keysDir, pkDir := runTestSetup(t)

	verifier, err := loadStore(roleVerifier, keysDir, "")
	if err != nil {
		t.Fatal(err)
	}
	prover, err := loadStore(roleProver, keysDir, pkDir)
	if err != nil {
		t.Fatal(err)
	}
	if verifier.manifestHash != prover.manifestHash {
		t.Fatal("roles loaded different manifests")
	}
	for _, id := range []string{normCircuitID, elgamalCircuitID} {
		v, p := verifier.byCircuit[id], prover.byCircuit[id]
		if v.vk == nil || v.pk != nil || v.cs != nil {
			t.Fatalf("verifier %s: must hold only the verifying key", id)
		}
		if p.pk == nil || p.cs == nil || p.vk != nil {
			t.Fatalf("prover %s: must hold only the proving key and circuit", id)
		}
		if v.vkHash != p.vkHash {
			t.Fatalf("%s: roles disagree on the pinned verifying key", id)
		}
	}
	if err := runSetup([]string{"--keys-dir", keysDir, "--pk-dir", pkDir}); err == nil {
		t.Fatal("setup overwrote existing pinned keys without --force")
	}
}

func TestLoadRefusesKeysThatDoNotMatchTheManifest(t *testing.T) {
	keysDir, pkDir := runTestSetup(t)
	flip := func(path string) {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		b[len(b)-1] ^= 0xff
		if err := os.WriteFile(path, b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	flip(filepath.Join(keysDir, "elgamal-2.vk"))
	if _, err := loadStore(roleVerifier, keysDir, ""); err == nil {
		t.Fatal("verifier loaded a tampered verifying key")
	}
	flip(filepath.Join(pkDir, "norm-4.pk"))
	if _, err := loadStore(roleProver, keysDir, pkDir); err == nil {
		t.Fatal("prover loaded a tampered proving key")
	}
	if _, err := loadStore(roleProver, keysDir, ""); err == nil {
		t.Fatal("prover started without a proving-key directory")
	}
}

func TestProofsVerifyAcrossSeparateProverAndVerifierStores(t *testing.T) {
	keysDir, pkDir := runTestSetup(t)
	prover, _ := loadStore(roleProver, keysDir, pkDir)
	verifier, _ := loadStore(roleVerifier, keysDir, "")
	sk := mustKeygen(t)
	pk := mulBase(sk)
	bound, ctx := big.NewInt(1_000_000), big.NewInt(3)
	glob, sums := plainTestGlobal(t, []int64{3})

	withStore(t, prover)
	cts, proof, err := elgamalProve(pk, sk, []int64{7}, glob, sums, bound, ctx) // 1 value, padded to n = 2
	if err != nil {
		t.Fatal(err)
	}
	if _, err := elgamalVerify(pk, cts, glob, bound, ctx, proof); err == nil {
		t.Fatal("prover role verified a proof")
	}
	store = verifier
	if ok, err := elgamalVerify(pk, cts, glob, bound, ctx, proof); err != nil || !ok {
		t.Fatalf("separate verifier rejected an honest padded proof: %v", err)
	}
	if _, _, err := elgamalProve(pk, sk, []int64{7}, glob, sums, bound, ctx); err == nil {
		t.Fatal("verifier role produced a proof")
	}
}

func normRequest(t *testing.T, values []int64, extra map[string]any) []byte {
	t.Helper()
	buf := make([]byte, 8*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint64(buf[8*i:], uint64(v))
	}
	body := map[string]any{"layer_name": "l", "weights_b64": base64.StdEncoding.EncodeToString(buf), "bound_sq": "1000000"}
	for k, v := range extra {
		body[k] = v
	}
	out, _ := json.Marshal(body)
	return out
}

func call(t *testing.T, handler http.HandlerFunc, body []byte) (int, map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body)))
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func TestNormProofIsPaddedAndPinnedToTheVerifyingKey(t *testing.T) {
	values := []int64{3, -4, 5} // padded to testNormN inside the service
	code, resp := call(t, proveHandler, normRequest(t, values, nil))
	if code != http.StatusOK {
		t.Fatalf("prove: %d %v", code, resp)
	}
	pinned := store.byCircuit[normCircuitID].vkHash
	if resp["vk_sha256"] != pinned || int(resp["circuit_n"].(float64)) != testNormN {
		t.Fatalf("prove response does not name the pinned key and size: %v", resp)
	}
	proof := resp["proof_b64"]

	code, resp = call(t, verifyHandler, normRequest(t, values, map[string]any{"proof_b64": proof, "vk_sha256": pinned}))
	if code != http.StatusOK || resp["verified"] != true {
		t.Fatalf("padded proof did not verify: %d %v", code, resp)
	}
	code, resp = call(t, verifyHandler, normRequest(t, values, map[string]any{"proof_b64": proof, "vk_sha256": "00" + pinned[2:]}))
	if code != http.StatusServiceUnavailable {
		t.Fatalf("declared verifying key mismatch was not refused loudly: %d %v", code, resp)
	}
	code, _ = call(t, proveHandler, normRequest(t, make([]int64, testNormN+1), nil))
	if code != http.StatusBadRequest {
		t.Fatalf("input longer than the fixed circuit size was accepted: %d", code)
	}
}
