package main

// Pinned Groth16 keys (audit/setup.md, Step 6).
//
// Keys are produced once by `gnark_service setup` and never at runtime:
//
//	keys/manifest.json      committed: circuit, fixed size, constraint count,
//	                        SHA-256 of every verifying and proving key
//	keys/<circuit>-<n>.vk   committed verifying keys
//	<pk-dir>/<circuit>-<n>.pk  proving keys, kept in a local cache outside git
//
// `gnark_service serve --role verifier` loads only verifying keys and serves
// only verification; `--role prover` loads only proving keys and serves only
// proving. Every key is checked against the manifest hash before use, and a
// verification request that declares a different verifying key is refused.
//
// Each circuit has one fixed size. Shorter inputs are padded inside the
// service with public, deterministic zero slots, so the key set is small and
// stable across models.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/consensys/gnark"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

const (
	normCircuitID    = "norm"
	elgamalCircuitID = "elgamal"
	manifestFile     = "manifest.json"
	roleProver       = "prover"
	roleVerifier     = "verifier"
)

type circuitKeys struct {
	n      int
	cs     constraint.ConstraintSystem // prover only
	pk     groth16.ProvingKey          // prover only
	vk     groth16.VerifyingKey        // verifier only
	vkHash string
}

type keyStore struct {
	role         string
	manifestHash string
	byCircuit    map[string]*circuitKeys
}

// store holds the keys this process was started with. Nil means none loaded.
var store *keyStore

var errKeyMismatch = errors.New("verifying key mismatch")

func (s *keyStore) keys(id string) (*circuitKeys, error) {
	if s == nil {
		return nil, fmt.Errorf("no keys loaded; start with `gnark_service serve`")
	}
	k, ok := s.byCircuit[id]
	if !ok {
		return nil, fmt.Errorf("%s has no key for circuit %q; keys come only from `gnark_service setup`", s.role, id)
	}
	return k, nil
}

// checkVK fails loudly when a request declares a verifying key other than the pinned one.
func (k *circuitKeys) checkVK(declared string) error {
	if declared != k.vkHash {
		return fmt.Errorf("%w: request declares %q, pinned key is %q", errKeyMismatch, declared, k.vkHash)
	}
	return nil
}

type manifestEntry struct {
	Circuit     string `json:"circuit"`
	N           int    `json:"n"`
	Constraints int    `json:"constraints"`
	VKFile      string `json:"vk_file"`
	VKSHA256    string `json:"vk_sha256"`
	PKFile      string `json:"pk_file"`
	PKSHA256    string `json:"pk_sha256"`
}

type keyManifest struct {
	Version   int             `json:"version"`
	Curve     string          `json:"curve"`
	Backend   string          `json:"backend"`
	Gnark     string          `json:"gnark_version"`
	CreatedAt string          `json:"created_at"`
	Setup     string          `json:"setup"`
	Circuits  []manifestEntry `json:"circuits"`
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func compileCircuit(id string, n int) (constraint.ConstraintSystem, error) {
	var circuit frontend.Circuit
	switch id {
	case normCircuitID:
		circuit = &proofCircuit{Weights: make([]frontend.Variable, n)}
	case elgamalCircuitID:
		if n > maxElgamalChunk {
			return nil, fmt.Errorf("elgamal chunk size %d exceeds %d", n, maxElgamalChunk)
		}
		circuit = newElgamalCircuit(n)
	default:
		return nil, fmt.Errorf("unknown circuit %q", id)
	}
	if n <= 0 {
		return nil, fmt.Errorf("circuit size must be positive, got %d", n)
	}
	return frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
}

// runSetup implements `gnark_service setup`.
func runSetup(args []string) error {
	fs := flag.NewFlagSet("setup", flag.ContinueOnError)
	keysDir := fs.String("keys-dir", "keys", "directory for the manifest and verifying keys (committed)")
	pkDir := fs.String("pk-dir", "", "directory for proving keys (local cache, not committed)")
	normN := fs.Int("norm-n", 256, "fixed size of the norm circuit")
	elgamalN := fs.Int("elgamal-n", 128, "fixed size of the ElGamal circuit")
	force := fs.Bool("force", false, "replace existing keys (invalidates proofs made with the old keys)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *pkDir == "" {
		return fmt.Errorf("--pk-dir is required")
	}
	manifestPath := filepath.Join(*keysDir, manifestFile)
	if _, err := os.Stat(manifestPath); err == nil && !*force {
		return fmt.Errorf("%s exists; pass --force to replace the pinned keys", manifestPath)
	}
	if err := os.MkdirAll(*keysDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(*pkDir, 0o700); err != nil {
		return err
	}

	m := keyManifest{
		Version:   1,
		Curve:     "BN254",
		Backend:   "groth16",
		Gnark:     gnark.Version.String(),
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		Setup: "single-party: one run of groth16.Setup by the operator of `gnark_service setup`. " +
			"The setup randomness existed in that process's memory and is not recoverable from these files, " +
			"but nothing proves it was destroyed. See audit/setup.md.",
	}
	for _, c := range []struct {
		id string
		n  int
	}{{normCircuitID, *normN}, {elgamalCircuitID, *elgamalN}} {
		start := time.Now()
		cs, err := compileCircuit(c.id, c.n)
		if err != nil {
			return err
		}
		pk, vk, err := groth16.Setup(cs)
		if err != nil {
			return err
		}
		var vkBuf, pkBuf bytes.Buffer
		if _, err := vk.WriteTo(&vkBuf); err != nil {
			return err
		}
		if _, err := pk.WriteRawTo(&pkBuf); err != nil {
			return err
		}
		entry := manifestEntry{
			Circuit:     c.id,
			N:           c.n,
			Constraints: cs.GetNbConstraints(),
			VKFile:      fmt.Sprintf("%s-%d.vk", c.id, c.n),
			VKSHA256:    sha256Hex(vkBuf.Bytes()),
			PKFile:      fmt.Sprintf("%s-%d.pk", c.id, c.n),
			PKSHA256:    sha256Hex(pkBuf.Bytes()),
		}
		if err := os.WriteFile(filepath.Join(*keysDir, entry.VKFile), vkBuf.Bytes(), 0o644); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*pkDir, entry.PKFile), pkBuf.Bytes(), 0o600); err != nil {
			return err
		}
		m.Circuits = append(m.Circuits, entry)
		log.Printf("setup %s n=%d constraints=%d vk=%s took=%s", c.id, c.n, entry.Constraints, entry.VKSHA256[:12], time.Since(start))
	}
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(manifestPath, append(out, '\n'), 0o644)
}

// loadStore loads only the keys the role needs, each checked against the manifest.
func loadStore(role, keysDir, pkDir string) (*keyStore, error) {
	raw, err := os.ReadFile(filepath.Join(keysDir, manifestFile))
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m keyManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	if m.Curve != "BN254" || m.Backend != "groth16" {
		return nil, fmt.Errorf("manifest is for %s/%s, not BN254/groth16", m.Curve, m.Backend)
	}
	s := &keyStore{role: role, manifestHash: sha256Hex(raw), byCircuit: map[string]*circuitKeys{}}

	for _, e := range m.Circuits {
		if _, dup := s.byCircuit[e.Circuit]; dup {
			return nil, fmt.Errorf("manifest lists circuit %q more than once; each circuit has one fixed size", e.Circuit)
		}
		k := &circuitKeys{n: e.N, vkHash: e.VKSHA256}
		switch role {
		case roleVerifier:
			b, err := os.ReadFile(filepath.Join(keysDir, e.VKFile))
			if err != nil {
				return nil, err
			}
			if got := sha256Hex(b); got != e.VKSHA256 {
				return nil, fmt.Errorf("%s: SHA-256 %s does not match manifest %s", e.VKFile, got, e.VKSHA256)
			}
			vk := groth16.NewVerifyingKey(ecc.BN254)
			if _, err := vk.ReadFrom(bytes.NewReader(b)); err != nil {
				return nil, fmt.Errorf("%s: %w", e.VKFile, err)
			}
			k.vk = vk
		case roleProver:
			if pkDir == "" {
				return nil, fmt.Errorf("prover role requires --pk-dir")
			}
			b, err := os.ReadFile(filepath.Join(pkDir, e.PKFile))
			if err != nil {
				return nil, fmt.Errorf("%w (restore the proving-key cache, or rerun setup, which replaces the pinned verifying keys)", err)
			}
			if got := sha256Hex(b); got != e.PKSHA256 {
				return nil, fmt.Errorf("%s: SHA-256 %s does not match manifest %s; it belongs to a different setup", e.PKFile, got, e.PKSHA256)
			}
			pk := groth16.NewProvingKey(ecc.BN254)
			if _, err := pk.UnsafeReadFrom(bytes.NewReader(b)); err != nil {
				return nil, fmt.Errorf("%s: %w", e.PKFile, err)
			}
			cs, err := compileCircuit(e.Circuit, e.N)
			if err != nil {
				return nil, err
			}
			if cs.GetNbConstraints() != e.Constraints {
				return nil, fmt.Errorf("circuit %s n=%d compiles to %d constraints, manifest records %d; the circuit changed since setup",
					e.Circuit, e.N, cs.GetNbConstraints(), e.Constraints)
			}
			k.pk, k.cs = pk, cs
		default:
			return nil, fmt.Errorf("unknown role %q (want %s or %s)", role, roleProver, roleVerifier)
		}
		s.byCircuit[e.Circuit] = k
	}
	for _, id := range []string{normCircuitID, elgamalCircuitID} {
		if _, ok := s.byCircuit[id]; !ok {
			return nil, fmt.Errorf("manifest has no entry for circuit %q", id)
		}
	}
	return s, nil
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	circuits := map[string]map[string]any{}
	for id, k := range store.byCircuit {
		circuits[id] = map[string]any{"n": k.n, "vk_sha256": k.vkHash}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":          "ok",
		"service":         "gnark-zkp",
		"role":            store.role,
		"manifest_sha256": store.manifestHash,
		"circuits":        circuits,
	})
}

// runServe implements `gnark_service serve --role prover|verifier`.
func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	role := fs.String("role", "", "prover or verifier")
	keysDir := fs.String("keys-dir", "keys", "directory with the manifest and verifying keys")
	pkDir := fs.String("pk-dir", "", "directory with proving keys (prover role)")
	port := fs.String("port", os.Getenv("ZKP_SERVICE_PORT"), "listen port")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *port == "" {
		*port = "9000"
	}
	s, err := loadStore(*role, *keysDir, *pkDir)
	if err != nil {
		return err
	}
	store = s

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	switch *role {
	case roleProver:
		mux.HandleFunc("/prove", proveHandler)
		registerElgamalProverRoutes(mux)
	case roleVerifier:
		mux.HandleFunc("/verify", verifyHandler)
		mux.HandleFunc("/verify_light", verifyLightHandler)
		registerElgamalVerifierRoutes(mux)
	}
	log.Printf("gnark ZKP %s listening on :%s (manifest %s)", *role, *port, s.manifestHash[:12])
	return http.ListenAndServe(":"+*port, mux)
}
