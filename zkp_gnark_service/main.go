package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	cryptomimc "github.com/consensys/gnark-crypto/ecc/bn254/fr/mimc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	gnarkmimc "github.com/consensys/gnark/std/hash/mimc"
)

type proofRequest struct {
	LayerName  string `json:"layer_name"`
	WeightsB64 string `json:"weights_b64"`
	Shape      []int  `json:"shape"`
	Scale      string `json:"scale"`
	BoundSq    string `json:"bound_sq"`
	ProofB64   string `json:"proof_b64"`
	VKSHA256   string `json:"vk_sha256"`
}

// verifyLightRequest is the payload for /verify_light.
// No weights required — only public SNARK inputs.
type verifyLightRequest struct {
	LayerName string `json:"layer_name"`
	Shape     []int  `json:"shape"`    // number of real values; the circuit pads to its fixed size
	BoundSq   string `json:"bound_sq"` // public input: Σwᵢ² ≤ bound
	HashHex   string `json:"hash_hex"` // public input: MiMC hash of the padded vector
	ProofB64  string `json:"proof_b64"`
	VKSHA256  string `json:"vk_sha256"`
}

type proofResponse struct {
	ProofB64 string `json:"proof_b64,omitempty"`
	HashHex  string `json:"hash_hex,omitempty"`
	Verified bool   `json:"verified"`
	VKSHA256 string `json:"vk_sha256,omitempty"`
	CircuitN int    `json:"circuit_n,omitempty"`
	Error    string `json:"error,omitempty"`
}

type proofCircuit struct {
	Weights []frontend.Variable
	Bound   frontend.Variable `gnark:",public"`
	Hash    frontend.Variable `gnark:",public"`
}

// int64Offset maps a signed 64-bit witness into [0, 2^64) for its range check.
var int64Offset = new(big.Int).Lsh(big.NewInt(1), 63)

func (c *proofCircuit) Define(api frontend.API) error {
	hasher, _ := gnarkmimc.NewMiMC(api)
	sum := api.Sub(0, 0) // Start with zero
	for _, w := range c.Weights {
		// Each witness is a signed 64-bit integer, so Σw² ≤ n·2^126 can't wrap the
		// field. Without this, a prover whose hash isn't recomputed by the verifier
		// (/verify_light) could pick large field elements whose squares sum to a
		// small value modulo the field.
		api.ToBinary(api.Add(w, int64Offset), 64)
		hasher.Write(w)
		sq := api.Mul(w, w)
		sum = api.Add(sum, sq)
	}
	hash := hasher.Sum()
	api.AssertIsEqual(hash, c.Hash)
	api.AssertIsLessOrEqual(sum, c.Bound)
	return nil
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

// padWeights appends zeros up to the circuit's fixed size. Zeros add nothing
// to the norm, and the hash covers the padded vector on both sides.
func padWeights(weights []*big.Int, n int) ([]*big.Int, error) {
	if len(weights) == 0 || len(weights) > n {
		return nil, fmt.Errorf("%d values do not fit the fixed circuit size %d", len(weights), n)
	}
	padded := make([]*big.Int, n)
	copy(padded, weights)
	for i := len(weights); i < n; i++ {
		padded[i] = big.NewInt(0)
	}
	return padded, nil
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

// keyStatus maps key-store errors to HTTP status: a key mismatch is an
// infrastructure fault (503), anything else a server error.
func keyStatus(err error) int {
	if errors.Is(err, errKeyMismatch) {
		return http.StatusServiceUnavailable
	}
	return http.StatusInternalServerError
}

func proveHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req proofRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	k, err := store.keys(normCircuitID)
	if err != nil || k.pk == nil {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("this service cannot prove: %v", err))
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
	padded, err := padWeights(weights, k.n)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	hashVal, err := computeHash(padded)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	assignment := &proofCircuit{
		Weights: make([]frontend.Variable, k.n),
		Bound:   boundSq,
		Hash:    hashVal,
	}
	for i, wv := range padded {
		assignment.Weights[i] = wv
	}

	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	proof, err := groth16.Prove(k.cs, k.pk, witness)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, fmt.Errorf("statement not satisfied: %w", err))
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
		VKSHA256: k.vkHash,
		CircuitN: k.n,
	}

	log.Printf("prove layer=%s n=%d/%d took=%s", req.LayerName, len(weights), k.n, time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

func verifyHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req proofRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	k, err := store.keys(normCircuitID)
	if err == nil && k.vk == nil {
		err = fmt.Errorf("%s role cannot verify", store.role)
	}
	if err == nil {
		err = k.checkVK(req.VKSHA256)
	}
	if err != nil {
		writeError(w, keyStatus(err), err)
		return
	}
	weights, err := decodeWeights(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	padded, err := padWeights(weights, k.n)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	boundSq, err := parseBigInt(req.BoundSq)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	hashVal, err := computeHash(padded)
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
		Weights: make([]frontend.Variable, k.n),
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

	verified := groth16.Verify(proof, k.vk, publicWitness) == nil
	resp := proofResponse{Verified: verified}
	if !verified {
		resp.Error = "proof verification failed"
	}

	log.Printf("verify layer=%s n=%d/%d verified=%t took=%s", req.LayerName, len(weights), k.n, verified, time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

// verifyLightHandler — verify a Groth16 proof using only public inputs.
//
// The client sends the proof alongside the public inputs (hash_hex, bound_sq)
// embedded during proof generation. No weights are transmitted; the pinned
// verifying key for the norm circuit is used.
func verifyLightHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req verifyLightRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	k, err := store.keys(normCircuitID)
	if err == nil && k.vk == nil {
		err = fmt.Errorf("%s role cannot verify", store.role)
	}
	if err == nil {
		err = k.checkVK(req.VKSHA256)
	}
	if err != nil {
		writeError(w, keyStatus(err), err)
		return
	}

	if len(req.Shape) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("shape must not be empty"))
		return
	}
	n := 1
	for _, d := range req.Shape {
		n *= d
	}
	if n <= 0 || n > k.n {
		writeError(w, http.StatusBadRequest, fmt.Errorf("shape holds %d values; the circuit takes at most %d", n, k.n))
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

	assignment := &proofCircuit{
		Weights: make([]frontend.Variable, k.n),
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

	verified := groth16.Verify(proof, k.vk, publicWitness) == nil
	resp := proofResponse{Verified: verified}
	if !verified {
		resp.Error = "proof verification failed"
	}

	log.Printf("verify_light layer=%s n=%d/%d verified=%t took=%s",
		req.LayerName, n, k.n, verified, time.Since(start))
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

const usage = `usage:
  gnark_service setup --keys-dir DIR --pk-dir DIR [--norm-n 256] [--elgamal-n 128] [--force]
  gnark_service serve --role prover|verifier --keys-dir DIR [--pk-dir DIR] [--port PORT]
  gnark_service elgamal-keygen <secret.json> <public.json>`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "setup":
		err = runSetup(os.Args[2:])
	case "serve":
		err = runServe(os.Args[2:])
	case "elgamal-keygen":
		err = runElgamalKeygen(os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
	if err != nil {
		log.Fatal(err)
	}
}
