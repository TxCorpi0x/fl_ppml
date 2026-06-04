package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptomimc "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	gnarkmimc "github.com/consensys/gnark/std/hash/mimc"
)

type proofRequest struct {
	LayerName string `json:"layer_name"`
	WeightsB64 string `json:"weights_b64"`
	Shape []int `json:"shape"`
	Scale string `json:"scale"`
	BoundSq string `json:"bound_sq"`
	ProofB64 string `json:"proof_b64"`
}

// verifyLightRequest is the payload for /verify_light.
// No weights required — only public SNARK inputs.
type verifyLightRequest struct {
	LayerName string `json:"layer_name"`
	Shape     []int  `json:"shape"`    // used to derive circuit size n = ∏shape
	BoundSq   string `json:"bound_sq"` // public input: Σwᵢ² ≤ bound
	HashHex   string `json:"hash_hex"` // public input: MiMC_hash(w)
	ProofB64  string `json:"proof_b64"`
}

type proofResponse struct {
	ProofB64 string `json:"proof_b64"`
	HashHex string `json:"hash_hex"`
	Verified bool `json:"verified"`
	Error string `json:"error,omitempty"`
}

type circuitCache struct {
	cs constraint.ConstraintSystem
	pk groth16.ProvingKey
	vk groth16.VerifyingKey
}

var (
	cacheMu sync.Mutex
	cache = map[int]*circuitCache{}
)

type proofCircuit struct {
	Weights []frontend.Variable
	Bound frontend.Variable `gnark:",public"`
	Hash frontend.Variable `gnark:",public"`
}

func (c *proofCircuit) Define(api frontend.API) error {
	hasher, _ := gnarkmimc.NewMiMC(api)
	sum := api.Sub(0, 0)  // Start with zero
	for _, w := range c.Weights {
		hasher.Write(w)
		sq := api.Mul(w, w)
		sum = api.Add(sum, sq)
	}
	hash := hasher.Sum()
	api.AssertIsEqual(hash, c.Hash)
	api.AssertIsLessOrEqual(sum, c.Bound)
	return nil
}

func getCircuit(n int) (*circuitCache, error) {
	cacheMu.Lock()
	defer cacheMu.Unlock()

	if cached, ok := cache[n]; ok {
		return cached, nil
	}

	circuit := &proofCircuit{Weights: make([]frontend.Variable, n)}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		return nil, err
	}
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		return nil, err
	}
	cached := &circuitCache{cs: cs, pk: pk, vk: vk}
	cache[n] = cached
	return cached, nil
}

func decodeWeights(req proofRequest) ([]*big.Int, error) {
	data, err := base64.StdEncoding.DecodeString(req.WeightsB64)
	if err != nil {
		return nil, fmt.Errorf("decode weights: %w", err)
	}
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("invalid weights byte length")
	}
	count := len(data) / 8
	weights := make([]*big.Int, count)
	for i := 0; i < count; i++ {
		offset := i * 8
		raw := binary.LittleEndian.Uint64(data[offset : offset+8])
		val := int64(raw)
		weights[i] = big.NewInt(val)
	}
	return weights, nil
}

func computeHash(weights []*big.Int) (*big.Int, error) {
	h := cryptomimc.NewMiMC()
	for _, w := range weights {
		var e fr.Element
		e.SetBigInt(w)
		bytes := e.Bytes()
		_, err := h.Write(bytes[:])
		if err != nil {
			return nil, err
		}
	}
	sum := h.Sum(nil)
	var out fr.Element
	out.SetBytes(sum)
	return out.BigInt(new(big.Int)), nil
}

func parseBigInt(value string) (*big.Int, error) {
	clean := strings.TrimSpace(value)
	if clean == "" {
		return nil, fmt.Errorf("missing integer")
	}
	result, ok := new(big.Int).SetString(clean, 10)
	if !ok {
		return nil, fmt.Errorf("invalid integer: %s", value)
	}
	return result, nil
}

func proveHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req proofRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	weights, err := decodeWeights(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	boundSq, err := parseBigInt(req.BoundSq)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if boundSq.Sign() <= 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("bound_sq must be positive"))
		return
	}
	if boundSq.Cmp(ecc.BN254.ScalarField()) >= 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("bound_sq exceeds field size"))
		return
	}

	hashVal, err := computeHash(weights)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	cached, err := getCircuit(len(weights))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	assignment := &proofCircuit{
		Weights: make([]frontend.Variable, len(weights)),
		Bound: boundSq,
		Hash: hashVal,
	}
	for i, wv := range weights {
		assignment.Weights[i] = wv
	}

	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	proof, err := groth16.Prove(cached.cs, cached.pk, witness)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	// Serialize proof using gnark's native WriteTo (avoids gob interface mismatch)
	var buf bytes.Buffer
	if _, err := proof.WriteTo(&buf); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	resp := proofResponse{
		ProofB64: base64.StdEncoding.EncodeToString(buf.Bytes()),
		HashHex:  fmt.Sprintf("%x", hashVal),
	}

	log.Printf("prove layer=%s n=%d took=%s", req.LayerName, len(weights), time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

func verifyHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req proofRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	weights, err := decodeWeights(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	boundSq, err := parseBigInt(req.BoundSq)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	hashVal, err := computeHash(weights)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	cached, err := getCircuit(len(weights))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	proofBytes, err := base64.StdEncoding.DecodeString(req.ProofB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	
	// Deserialize proof using gnark's native ReadFrom to avoid gob interface mismatch
	proof := groth16.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("failed to decode proof: %w", err))
		return
	}

	// For verification we only need public inputs (Bound, Hash).
	// Private Weights must be non-nil; set to zero (value doesn't affect public witness).
	assignment := &proofCircuit{
		Weights: make([]frontend.Variable, len(weights)),
		Bound:   boundSq,
		Hash:    hashVal,
	}
	for i := range assignment.Weights {
		assignment.Weights[i] = big.NewInt(0)
	}
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	publicWitness, err := witness.Public()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	verified := groth16.Verify(proof, cached.vk, publicWitness) == nil
	resp := proofResponse{Verified: verified}
	if !verified {
		resp.Error = "proof verification failed"
	}

	log.Printf("verify layer=%s n=%d verified=%t took=%s", req.LayerName, len(weights), verified, time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

// verifyLightHandler — verify a Groth16 proof using only public inputs.
//
// The client (working under FHE) sends the proof alongside the committed
// public inputs (hash_hex, bound_sq) that were embedded during proof
// generation.  No plaintext weights are transmitted; the server re-uses the
// cached verifying key (keyed on circuit size n = ∏shape) and verifies.
//
// This is the zero-knowledge path: the server learns nothing about the
// weights beyond what the public inputs reveal.
func verifyLightHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req verifyLightRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	// Derive circuit size from shape.
	if len(req.Shape) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("shape must not be empty"))
		return
	}
	n := 1
	for _, d := range req.Shape {
		n *= d
	}
	if n <= 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid shape: n=%d", n))
		return
	}

	// Parse public inputs.
	boundSq, err := parseBigInt(req.BoundSq)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("bound_sq: %w", err))
		return
	}
	hashBytes, err := decodeHex(req.HashHex)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("hash_hex: %w", err))
		return
	}
	hashVal := new(big.Int).SetBytes(hashBytes)

	// Look up (or compile) the verifying key for this circuit size.
	cached, err := getCircuit(n)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	// Decode proof.
	proofBytes, err := base64.StdEncoding.DecodeString(req.ProofB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("proof_b64: %w", err))
		return
	}
	proof := groth16.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("failed to decode proof: %w", err))
		return
	}

	// Build public witness — private Weights field set to zero-length slice
	// so gnark only serialises the public inputs (Bound, Hash).
	assignment := &proofCircuit{
		Weights: make([]frontend.Variable, n),
		Bound:   boundSq,
		Hash:    hashVal,
	}
	for i := range assignment.Weights {
		assignment.Weights[i] = big.NewInt(0)
	}
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	publicWitness, err := witness.Public()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	verified := groth16.Verify(proof, cached.vk, publicWitness) == nil
	resp := proofResponse{Verified: verified}
	if !verified {
		resp.Error = "proof verification failed"
	}

	log.Printf("verify_light layer=%s n=%d verified=%t took=%s",
		req.LayerName, n, verified, time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

// decodeHex parses a hex string (with or without 0x prefix) into bytes.
func decodeHex(s string) ([]byte, error) {
	clean := strings.TrimPrefix(strings.TrimSpace(s), "0x")
	if len(clean)%2 != 0 {
		clean = "0" + clean
	}
	var buf []byte
	for i := 0; i < len(clean); i += 2 {
		var b byte
		_, err := fmt.Sscanf(clean[i:i+2], "%02x", &b)
		if err != nil {
			return nil, fmt.Errorf("invalid hex at pos %d: %w", i, err)
		}
		buf = append(buf, b)
	}
	return buf, nil
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, proofResponse{Verified: false, Error: err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(payload)
}

func main() {
	port := os.Getenv("ZKP_SERVICE_PORT")
	if port == "" {
		port = "9000"
	}

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "service": "gnark-zkp"})
	})
	http.HandleFunc("/prove", proveHandler)
	http.HandleFunc("/verify", verifyHandler)
	http.HandleFunc("/verify_light", verifyLightHandler)

	addr := ":" + port
	log.Printf("gnark ZKP service listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}
