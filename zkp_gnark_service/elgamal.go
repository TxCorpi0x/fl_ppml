package main

// Verifiable additive encryption for federated updates (audit/binding.md Design B + A).
//
// Each coordinate of a quantized update q (|q| < 2^(elgamalValueBits-1)) is
// offset-encoded as v = q + 2^(elgamalValueBits-1) and encrypted with
// exponential ElGamal on the BN254 twisted Edwards curve (BabyJubJub):
//
//	C1 = r·G,  C2 = v·G + r·PK
//
// One Groth16 proof per chunk shows, with the ciphertexts themselves as public
// inputs, that every C encrypts a v in range and that Σ(v − offset)² ≤ Bound.
// The server verifies against the ciphertexts it received, so a proof cannot be
// paired with a ciphertext of a different vector. Context (round, layer, chunk)
// is a public input so a proof cannot be replayed at another position or round.
//
// Not addressed here: trusted setup is still generated lazily in-process
// (audit/findings.md S1-08), and the bound applies to the submitted weights,
// not the update (S1-07).

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	edbn254 "github.com/consensys/gnark-crypto/ecc/bn254/twistededwards"
	tedwards "github.com/consensys/gnark-crypto/ecc/twistededwards"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/algebra/native/twistededwards"
)

const (
	elgamalValueBits = 18 // v ∈ [0, 2^18), q ∈ [-2^17, 2^17)
	elgamalPointSize = 32 // compressed twisted Edwards point
	elgamalCtSize    = 2 * elgamalPointSize
	bsgsTableBits    = 16
	maxElgamalChunk  = 4096
)

var (
	elgamalOffset = new(big.Int).Lsh(big.NewInt(1), elgamalValueBits-1)
	bn254R        = ecc.BN254.ScalarField()
)

// ── Circuit ──────────────────────────────────────────────────────────────────

type elgamalCircuit struct {
	PKX     frontend.Variable   `gnark:",public"`
	PKY     frontend.Variable   `gnark:",public"`
	Bound   frontend.Variable   `gnark:",public"`
	Context frontend.Variable   `gnark:",public"`
	C1X     []frontend.Variable `gnark:",public"`
	C1Y     []frontend.Variable `gnark:",public"`
	C2X     []frontend.Variable `gnark:",public"`
	C2Y     []frontend.Variable `gnark:",public"`
	Values  []frontend.Variable
	Rand    []frontend.Variable
}

func newElgamalCircuit(n int) *elgamalCircuit {
	return &elgamalCircuit{
		C1X: make([]frontend.Variable, n), C1Y: make([]frontend.Variable, n),
		C2X: make([]frontend.Variable, n), C2Y: make([]frontend.Variable, n),
		Values: make([]frontend.Variable, n), Rand: make([]frontend.Variable, n),
	}
}

func (c *elgamalCircuit) Define(api frontend.API) error {
	curve, err := twistededwards.NewEdCurve(api, tedwards.BN254)
	if err != nil {
		return err
	}
	base := curve.Params().Base
	g := twistededwards.Point{X: base[0], Y: base[1]}
	pk := twistededwards.Point{X: c.PKX, Y: c.PKY}
	curve.AssertIsOnCurve(pk)

	// Context takes part in no other constraint. A Groth16 public input that
	// appears in no constraint is not bound by the proof, so bind it here.
	_ = api.Mul(c.Context, c.Context)

	sum := frontend.Variable(0)
	for i := range c.Values {
		// Range check doubles as the scalar decomposition for v·G.
		bits := api.ToBinary(c.Values[i], elgamalValueBits)
		vg := twistededwards.Point{X: 0, Y: 1}
		for j := elgamalValueBits - 1; j >= 0; j-- {
			vg = curve.Double(vg)
			added := curve.Add(vg, g)
			vg = twistededwards.Point{
				X: api.Select(bits[j], added.X, vg.X),
				Y: api.Select(bits[j], added.Y, vg.Y),
			}
		}

		c1 := curve.ScalarMul(g, c.Rand[i])
		api.AssertIsEqual(c1.X, c.C1X[i])
		api.AssertIsEqual(c1.Y, c.C1Y[i])

		c2 := curve.Add(vg, curve.ScalarMul(pk, c.Rand[i]))
		api.AssertIsEqual(c2.X, c.C2X[i])
		api.AssertIsEqual(c2.Y, c.C2Y[i])

		q := api.Sub(c.Values[i], elgamalOffset)
		sum = api.Add(sum, api.Mul(q, q))
	}
	api.AssertIsLessOrEqual(sum, c.Bound)
	return nil
}

var (
	elgamalCacheMu sync.Mutex
	elgamalCache   = map[int]*circuitCache{}
)

func getElgamalCircuit(n int) (*circuitCache, error) {
	if n <= 0 || n > maxElgamalChunk {
		return nil, fmt.Errorf("chunk size %d outside [1, %d]", n, maxElgamalChunk)
	}
	elgamalCacheMu.Lock()
	defer elgamalCacheMu.Unlock()
	if cached, ok := elgamalCache[n]; ok {
		return cached, nil
	}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, newElgamalCircuit(n))
	if err != nil {
		return nil, err
	}
	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		return nil, err
	}
	cached := &circuitCache{cs: cs, pk: pk, vk: vk}
	elgamalCache[n] = cached
	return cached, nil
}

// ── Native curve helpers ─────────────────────────────────────────────────────

func edParams() edbn254.CurveParams { return edbn254.GetEdwardsCurve() }

func identityPoint() edbn254.PointAffine {
	var zero, one fr.Element
	one.SetOne()
	return edbn254.NewPointAffine(zero, one)
}

func mulBase(s *big.Int) edbn254.PointAffine {
	params := edParams()
	var p edbn254.PointAffine
	p.ScalarMultiplication(&params.Base, s)
	return p
}

func randomScalar() (*big.Int, error) {
	params := edParams()
	return rand.Int(rand.Reader, &params.Order)
}

// decodePoint accepts only canonical compressed encodings of on-curve points.
func decodePoint(b []byte) (edbn254.PointAffine, error) {
	var p edbn254.PointAffine
	if len(b) != elgamalPointSize {
		return p, fmt.Errorf("point must be %d bytes", elgamalPointSize)
	}
	if _, err := p.SetBytes(b); err != nil {
		return p, fmt.Errorf("invalid point: %w", err)
	}
	if !p.IsOnCurve() {
		return p, fmt.Errorf("point not on curve")
	}
	if enc := p.Bytes(); !bytes.Equal(enc[:], b) {
		return p, fmt.Errorf("non-canonical point encoding")
	}
	return p, nil
}

func decodePublicKey(b []byte) (edbn254.PointAffine, error) {
	pk, err := decodePoint(b)
	if err != nil {
		return pk, err
	}
	params := edParams()
	var check edbn254.PointAffine
	check.ScalarMultiplication(&pk, &params.Order)
	if !check.IsZero() || pk.IsZero() {
		return pk, fmt.Errorf("public key not in prime-order subgroup")
	}
	return pk, nil
}

type elgamalCiphertext struct{ C1, C2 edbn254.PointAffine }

func decodeCiphertexts(b []byte) ([]elgamalCiphertext, error) {
	if len(b) == 0 || len(b)%elgamalCtSize != 0 {
		return nil, fmt.Errorf("ciphertext length %d not a positive multiple of %d", len(b), elgamalCtSize)
	}
	out := make([]elgamalCiphertext, len(b)/elgamalCtSize)
	for i := range out {
		off := i * elgamalCtSize
		c1, err := decodePoint(b[off : off+elgamalPointSize])
		if err != nil {
			return nil, fmt.Errorf("ciphertext %d C1: %w", i, err)
		}
		c2, err := decodePoint(b[off+elgamalPointSize : off+elgamalCtSize])
		if err != nil {
			return nil, fmt.Errorf("ciphertext %d C2: %w", i, err)
		}
		out[i] = elgamalCiphertext{c1, c2}
	}
	return out, nil
}

func encodeCiphertexts(cts []elgamalCiphertext) []byte {
	out := make([]byte, 0, len(cts)*elgamalCtSize)
	for _, ct := range cts {
		b1, b2 := ct.C1.Bytes(), ct.C2.Bytes()
		out = append(out, b1[:]...)
		out = append(out, b2[:]...)
	}
	return out
}

func coord(e fr.Element) *big.Int { return e.BigInt(new(big.Int)) }

// canonicalFieldInt parses a decimal integer in [0, r).
func canonicalFieldInt(s, name string) (*big.Int, error) {
	v, err := parseBigInt(s)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if v.Sign() < 0 || v.Cmp(bn254R) >= 0 {
		return nil, fmt.Errorf("%s must be in [0, r)", name)
	}
	return v, nil
}

// ── Protocol operations ──────────────────────────────────────────────────────

func elgamalKeygen() (*big.Int, edbn254.PointAffine, error) {
	sk, err := randomScalar()
	if err != nil {
		return nil, edbn254.PointAffine{}, err
	}
	for sk.Sign() == 0 {
		if sk, err = randomScalar(); err != nil {
			return nil, edbn254.PointAffine{}, err
		}
	}
	return sk, mulBase(sk), nil
}

func checkValue(q int64, i int) error {
	limit := elgamalOffset.Int64()
	if q < -limit || q >= limit {
		return fmt.Errorf("value %d at index %d outside [-%d, %d)", q, i, limit, limit)
	}
	return nil
}

// encryptWith returns (r·G, (q + offset)·G + r·PK).
func encryptWith(pk edbn254.PointAffine, q int64, r *big.Int) elgamalCiphertext {
	v := new(big.Int).Add(big.NewInt(q), elgamalOffset)
	var rpk, c2 edbn254.PointAffine
	rpk.ScalarMultiplication(&pk, r)
	vg := mulBase(v)
	c2.Add(&vg, &rpk)
	return elgamalCiphertext{C1: mulBase(r), C2: c2}
}

// elgamalEncrypt encrypts quantized values under pk with fresh randomness,
// returning the randomness so the client can later prove statements about
// exactly these ciphertexts (commit–challenge sampling, audit/sampling.md).
func elgamalEncrypt(pk edbn254.PointAffine, qs []int64) ([]elgamalCiphertext, []*big.Int, error) {
	cts := make([]elgamalCiphertext, len(qs))
	rands := make([]*big.Int, len(qs))
	for i, q := range qs {
		if err := checkValue(q, i); err != nil {
			return nil, nil, err
		}
		r, err := randomScalar()
		if err != nil {
			return nil, nil, err
		}
		rands[i] = r
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i := range qs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			cts[i] = encryptWith(pk, qs[i], rands[i])
		}(i)
	}
	wg.Wait()
	return cts, rands, nil
}

// elgamalProve encrypts quantized values under fresh randomness and proves the chunk statement.
func elgamalProve(pk edbn254.PointAffine, qs []int64, bound, context *big.Int) ([]elgamalCiphertext, []byte, error) {
	rands := make([]*big.Int, len(qs))
	for i := range qs {
		r, err := randomScalar()
		if err != nil {
			return nil, nil, err
		}
		rands[i] = r
	}
	return elgamalProveWith(pk, qs, rands, bound, context)
}

// elgamalProveWith proves the chunk statement for the ciphertexts determined by
// (qs, rands). The returned ciphertexts are recomputed, so a proof made with
// the wrong values or randomness will not verify against a stored commitment.
func elgamalProveWith(pk edbn254.PointAffine, qs []int64, rands []*big.Int, bound, context *big.Int) ([]elgamalCiphertext, []byte, error) {
	n := len(qs)
	if len(rands) != n {
		return nil, nil, fmt.Errorf("%d values but %d randomness scalars", n, len(rands))
	}
	cached, err := getElgamalCircuit(n)
	if err != nil {
		return nil, nil, err
	}
	params := edParams()
	assignment := newElgamalCircuit(n)
	assignment.PKX, assignment.PKY = coord(pk.X), coord(pk.Y)
	assignment.Bound, assignment.Context = bound, context

	cts := make([]elgamalCiphertext, n)
	for i, q := range qs {
		if err := checkValue(q, i); err != nil {
			return nil, nil, err
		}
		r := rands[i]
		if r == nil || r.Sign() < 0 || r.Cmp(&params.Order) >= 0 {
			return nil, nil, fmt.Errorf("randomness at index %d outside [0, order)", i)
		}
		cts[i] = encryptWith(pk, q, r)

		assignment.Values[i], assignment.Rand[i] = new(big.Int).Add(big.NewInt(q), elgamalOffset), r
		assignment.C1X[i], assignment.C1Y[i] = coord(cts[i].C1.X), coord(cts[i].C1.Y)
		assignment.C2X[i], assignment.C2Y[i] = coord(cts[i].C2.X), coord(cts[i].C2.Y)
	}

	w, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		return nil, nil, err
	}
	proof, err := groth16.Prove(cached.cs, cached.pk, w)
	if err != nil {
		return nil, nil, fmt.Errorf("statement not satisfied: %w", err)
	}
	var buf bytes.Buffer
	if _, err := proof.WriteTo(&buf); err != nil {
		return nil, nil, err
	}
	return cts, buf.Bytes(), nil
}

// elgamalVerify checks a chunk proof against ciphertexts the verifier holds.
func elgamalVerify(pk edbn254.PointAffine, cts []elgamalCiphertext, bound, context *big.Int, proofBytes []byte) (bool, error) {
	n := len(cts)
	cached, err := getElgamalCircuit(n)
	if err != nil {
		return false, err
	}
	proof := groth16.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		return false, fmt.Errorf("decode proof: %w", err)
	}
	assignment := newElgamalCircuit(n)
	assignment.PKX, assignment.PKY = coord(pk.X), coord(pk.Y)
	assignment.Bound, assignment.Context = bound, context
	for i, ct := range cts {
		assignment.C1X[i], assignment.C1Y[i] = coord(ct.C1.X), coord(ct.C1.Y)
		assignment.C2X[i], assignment.C2Y[i] = coord(ct.C2.X), coord(ct.C2.Y)
		assignment.Values[i], assignment.Rand[i] = 0, 0
	}
	w, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		return false, err
	}
	public, err := w.Public()
	if err != nil {
		return false, err
	}
	return groth16.Verify(proof, cached.vk, public) == nil, nil
}

// elgamalAggregate returns Σ weight_i · ct_i coordinate-wise.
func elgamalAggregate(clients [][]elgamalCiphertext, weights []int64) ([]elgamalCiphertext, error) {
	if len(clients) == 0 || len(clients) != len(weights) {
		return nil, fmt.Errorf("need one weight per client ciphertext")
	}
	n := len(clients[0])
	out := make([]elgamalCiphertext, n)
	for i := range out {
		out[i] = elgamalCiphertext{identityPoint(), identityPoint()}
	}
	for k, cts := range clients {
		if len(cts) != n {
			return nil, fmt.Errorf("client %d has %d ciphertexts, expected %d", k, len(cts), n)
		}
		if weights[k] <= 0 {
			return nil, fmt.Errorf("weight %d must be positive", weights[k])
		}
		wk := big.NewInt(weights[k])
		for i, ct := range cts {
			var a, b edbn254.PointAffine
			a.ScalarMultiplication(&ct.C1, wk)
			b.ScalarMultiplication(&ct.C2, wk)
			out[i].C1.Add(&out[i].C1, &a)
			out[i].C2.Add(&out[i].C2, &b)
		}
	}
	return out, nil
}

var (
	bsgsOnce  sync.Once
	bsgsTable map[[elgamalPointSize]byte]int32
	bsgsStep  edbn254.PointAffine
)

func buildBSGS() {
	params := edParams()
	half := int64(1) << (bsgsTableBits - 1)
	bsgsTable = make(map[[elgamalPointSize]byte]int32, 1<<bsgsTableBits)
	p := mulBase(big.NewInt(half))
	p.Neg(&p)
	for i := -half; i < half; i++ {
		bsgsTable[p.Bytes()] = int32(i)
		p.Add(&p, &params.Base)
	}
	bsgsStep = mulBase(big.NewInt(1 << bsgsTableBits))
}

// discreteLog finds m with m·G = target and |m| ≤ maxAbs, searching outward from 0.
func discreteLog(target edbn254.PointAffine, maxAbs int64) (int64, bool) {
	bsgsOnce.Do(buildBSGS)
	step := int64(1) << bsgsTableBits
	maxJ := maxAbs/step + 1
	up, down := target, target
	for j := int64(0); j <= maxJ; j++ {
		if i, ok := bsgsTable[up.Bytes()]; ok {
			return j*step + int64(i), true
		}
		if j > 0 {
			if i, ok := bsgsTable[down.Bytes()]; ok {
				return -j*step + int64(i), true
			}
		}
		var negStep edbn254.PointAffine
		negStep.Neg(&bsgsStep)
		up.Add(&up, &negStep)
		down.Add(&down, &bsgsStep)
	}
	return 0, false
}

// elgamalDecrypt returns m_i = Σ w_k q_k for each coordinate, given offsetTotal = Σ w_k · 2^17.
func elgamalDecrypt(sk *big.Int, cts []elgamalCiphertext, offsetTotal *big.Int, maxAbs int64) ([]int64, error) {
	negOffset := mulBase(offsetTotal)
	negOffset.Neg(&negOffset)
	out := make([]int64, len(cts))
	errs := make([]error, len(cts))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	for i := range cts {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			var skC1, target edbn254.PointAffine
			skC1.ScalarMultiplication(&cts[i].C1, sk)
			skC1.Neg(&skC1)
			target.Add(&cts[i].C2, &skC1)
			target.Add(&target, &negOffset)
			m, ok := discreteLog(target, maxAbs)
			if !ok {
				errs[i] = fmt.Errorf("coordinate %d: plaintext outside ±%d", i, maxAbs)
				return
			}
			out[i] = m
		}(i)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ── HTTP handlers ────────────────────────────────────────────────────────────

type elgamalProveRequest struct {
	PK        string `json:"pk"`
	ValuesB64 string `json:"values_b64"` // little-endian int64 quantized q
	BoundSq   string `json:"bound_sq"`
	Context   string `json:"context"`
}

type elgamalEncryptRequest struct {
	PK        string `json:"pk"`
	ValuesB64 string `json:"values_b64"` // little-endian int64 quantized q
}

type elgamalProveWithRequest struct {
	PK        string `json:"pk"`
	ValuesB64 string `json:"values_b64"`
	RandB64   string `json:"rand_b64"` // 32-byte big-endian scalars, one per value
	BoundSq   string `json:"bound_sq"`
	Context   string `json:"context"`
}

const scalarSize = 32

func encodeScalars(rs []*big.Int) []byte {
	out := make([]byte, len(rs)*scalarSize)
	for i, r := range rs {
		r.FillBytes(out[i*scalarSize : (i+1)*scalarSize])
	}
	return out
}

func decodeScalars(b64 string, n int) ([]*big.Int, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	if len(raw) != n*scalarSize {
		return nil, fmt.Errorf("expected %d randomness bytes, got %d", n*scalarSize, len(raw))
	}
	out := make([]*big.Int, n)
	for i := range out {
		out[i] = new(big.Int).SetBytes(raw[i*scalarSize : (i+1)*scalarSize])
	}
	return out, nil
}

func elgamalEncryptHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req elgamalEncryptRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	pkBytes, err := decodeHex(req.PK)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("pk: %w", err))
		return
	}
	pk, err := decodePublicKey(pkBytes)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("pk: %w", err))
		return
	}
	qs, err := decodeInt64s(req.ValuesB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("values_b64: %w", err))
		return
	}
	cts, rands, err := elgamalEncrypt(pk, qs)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err)
		return
	}
	log.Printf("elgamal encrypt n=%d took=%s", len(qs), time.Since(start))
	writeJSON(w, http.StatusOK, map[string]string{
		"ct_b64":   base64.StdEncoding.EncodeToString(encodeCiphertexts(cts)),
		"rand_b64": base64.StdEncoding.EncodeToString(encodeScalars(rands)),
	})
}

func elgamalProveWithHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req elgamalProveWithRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	pk, bound, ctx, err := decodePublicInputs(req.PK, req.BoundSq, req.Context)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	qs, err := decodeInt64s(req.ValuesB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("values_b64: %w", err))
		return
	}
	rands, err := decodeScalars(req.RandB64, len(qs))
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("rand_b64: %w", err))
		return
	}
	cts, proof, err := elgamalProveWith(pk, qs, rands, bound, ctx)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err)
		return
	}
	log.Printf("elgamal prove_with n=%d took=%s", len(qs), time.Since(start))
	writeJSON(w, http.StatusOK, map[string]string{
		"ct_b64":    base64.StdEncoding.EncodeToString(encodeCiphertexts(cts)),
		"proof_b64": base64.StdEncoding.EncodeToString(proof),
	})
}

type elgamalVerifyRequest struct {
	PK       string `json:"pk"`
	CtB64    string `json:"ct_b64"`
	BoundSq  string `json:"bound_sq"`
	Context  string `json:"context"`
	ProofB64 string `json:"proof_b64"`
}

type elgamalAggregateRequest struct {
	CtsB64  []string `json:"cts_b64"`
	Weights []int64  `json:"weights"`
}

type elgamalDecryptRequest struct {
	SK          string `json:"sk"`
	CtB64       string `json:"ct_b64"`
	OffsetTotal string `json:"offset_total"`
	MaxAbs      int64  `json:"max_abs"`
}

func decodeJSON(r *http.Request, v any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}

func decodeInt64s(b64 string) ([]int64, error) {
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data)%8 != 0 {
		return nil, fmt.Errorf("values must be a positive multiple of 8 bytes")
	}
	out := make([]int64, len(data)/8)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out, nil
}

func decodePublicInputs(pkHex, boundSq, context string) (edbn254.PointAffine, *big.Int, *big.Int, error) {
	pkBytes, err := decodeHex(pkHex)
	if err != nil {
		return edbn254.PointAffine{}, nil, nil, fmt.Errorf("pk: %w", err)
	}
	pk, err := decodePublicKey(pkBytes)
	if err != nil {
		return pk, nil, nil, fmt.Errorf("pk: %w", err)
	}
	bound, err := canonicalFieldInt(boundSq, "bound_sq")
	if err != nil {
		return pk, nil, nil, err
	}
	if bound.Sign() <= 0 {
		return pk, nil, nil, fmt.Errorf("bound_sq must be positive")
	}
	ctx, err := canonicalFieldInt(context, "context")
	if err != nil {
		return pk, nil, nil, err
	}
	return pk, bound, ctx, nil
}

func elgamalProveHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req elgamalProveRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	pk, bound, ctx, err := decodePublicInputs(req.PK, req.BoundSq, req.Context)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	qs, err := decodeInt64s(req.ValuesB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("values_b64: %w", err))
		return
	}
	cts, proof, err := elgamalProve(pk, qs, bound, ctx)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err)
		return
	}
	log.Printf("elgamal prove n=%d took=%s", len(qs), time.Since(start))
	writeJSON(w, http.StatusOK, map[string]string{
		"ct_b64":    base64.StdEncoding.EncodeToString(encodeCiphertexts(cts)),
		"proof_b64": base64.StdEncoding.EncodeToString(proof),
	})
}

func elgamalVerifyHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req elgamalVerifyRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	pk, bound, ctx, err := decodePublicInputs(req.PK, req.BoundSq, req.Context)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	ctBytes, err := base64.StdEncoding.DecodeString(req.CtB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("ct_b64: %w", err))
		return
	}
	cts, err := decodeCiphertexts(ctBytes)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	proof, err := base64.StdEncoding.DecodeString(req.ProofB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("proof_b64: %w", err))
		return
	}
	ok, err := elgamalVerify(pk, cts, bound, ctx, proof)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	resp := proofResponse{Verified: ok}
	if !ok {
		resp.Error = "proof verification failed"
	}
	log.Printf("elgamal verify n=%d verified=%t took=%s", len(cts), ok, time.Since(start))
	writeJSON(w, http.StatusOK, resp)
}

func elgamalAggregateHandler(w http.ResponseWriter, r *http.Request) {
	var req elgamalAggregateRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	clients := make([][]elgamalCiphertext, len(req.CtsB64))
	for k, b64 := range req.CtsB64 {
		raw, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Errorf("client %d: %w", k, err))
			return
		}
		if clients[k], err = decodeCiphertexts(raw); err != nil {
			writeError(w, http.StatusBadRequest, fmt.Errorf("client %d: %w", k, err))
			return
		}
	}
	sum, err := elgamalAggregate(clients, req.Weights)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"ct_b64": base64.StdEncoding.EncodeToString(encodeCiphertexts(sum)),
	})
}

func elgamalDecryptHandler(w http.ResponseWriter, r *http.Request) {
	var req elgamalDecryptRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	sk, err := canonicalFieldInt(req.SK, "sk")
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	offset, err := canonicalFieldInt(req.OffsetTotal, "offset_total")
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.MaxAbs <= 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("max_abs must be positive"))
		return
	}
	raw, err := base64.StdEncoding.DecodeString(req.CtB64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("ct_b64: %w", err))
		return
	}
	cts, err := decodeCiphertexts(raw)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	values, err := elgamalDecrypt(sk, cts, offset, req.MaxAbs)
	if err != nil {
		writeError(w, http.StatusUnprocessableEntity, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string][]int64{"values": values})
}

func elgamalInfoHandler(w http.ResponseWriter, r *http.Request) {
	n, err := strconv.Atoi(r.URL.Query().Get("n"))
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("n: %w", err))
		return
	}
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, newElgamalCircuit(n))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	legacy, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &proofCircuit{Weights: make([]frontend.Variable, n)})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{
		"n":                          n,
		"elgamal_constraints":        cs.GetNbConstraints(),
		"elgamal_public_inputs":      cs.GetNbPublicVariables(),
		"unbound_norm_constraints":   legacy.GetNbConstraints(),
		"unbound_norm_public_inputs": legacy.GetNbPublicVariables(),
	})
}

func registerElgamalRoutes() {
	http.HandleFunc("/elgamal/prove", elgamalProveHandler)
	http.HandleFunc("/elgamal/encrypt", elgamalEncryptHandler)
	http.HandleFunc("/elgamal/prove_with", elgamalProveWithHandler)
	http.HandleFunc("/elgamal/verify", elgamalVerifyHandler)
	http.HandleFunc("/elgamal/aggregate", elgamalAggregateHandler)
	http.HandleFunc("/elgamal/decrypt", elgamalDecryptHandler)
	http.HandleFunc("/elgamal/info", elgamalInfoHandler)
}

// runElgamalKeygen implements `gnark_service elgamal-keygen <secret.json> <public.json>`.
// The secret file holds the shared client decryption key; the public file is
// all the server needs.
func runElgamalKeygen(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: gnark_service elgamal-keygen <secret.json> <public.json>")
	}
	secretPath, publicPath := args[0], args[1]
	for _, p := range args {
		if _, err := os.Stat(p); err == nil {
			return fmt.Errorf("%s already exists; refusing to overwrite key material", p)
		}
	}
	sk, pk, err := elgamalKeygen()
	if err != nil {
		return err
	}
	pkBytes := pk.Bytes()
	pkHex := fmt.Sprintf("%x", pkBytes[:])
	secret, _ := json.MarshalIndent(map[string]string{"sk": sk.String(), "pk": pkHex}, "", "  ")
	public, _ := json.MarshalIndent(map[string]string{"pk": pkHex}, "", "  ")
	if err := os.WriteFile(secretPath, secret, 0o600); err != nil {
		return err
	}
	return os.WriteFile(publicPath, public, 0o644)
}
