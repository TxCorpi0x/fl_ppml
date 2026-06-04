// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title  FLLedger
 * @notice On-chain audit trail for Privacy-Preserving Federated Learning.
 *
 * @dev    Two write operations are supported:
 *
 *           commitModel()  — called by the server aggregator after each FL
 *                            round to record the SHA-256 hash of the new
 *                            global model and the per-client update hashes.
 *
 *           anchorProofs() — called after ZKP verification to permanently
 *                            record the hashes of accepted gnark proofs,
 *                            providing a tamper-evident proof audit trail.
 *
 *         Both functions emit indexed events so off-chain services can
 *         reconstruct the full training history from logs alone.
 *
 * Deployment (Hardhat / Foundry):
 *   npx hardhat run scripts/deploy_fl_ledger.js --network localhost
 *   forge create src/FLLedger.sol:FLLedger --rpc-url http://127.0.0.1:8545
 *
 * Once deployed, set FL_CHAIN_CONTRACT_ADDR to the contract address and
 * FL_CHAIN_BACKEND=web3 before starting the FL server.
 */
contract FLLedger {

    // ── Events ────────────────────────────────────────────────────────────────

    /**
     * @notice Emitted when the aggregated model for a round is committed.
     * @param round      FL training round number (1-based).
     * @param modelHash  SHA-256 of the aggregated model parameters.
     * @param numClients Number of client updates included in this round.
     * @param timestamp  Block timestamp of the commit.
     */
    event ModelCommitted(
        uint256 indexed round,
        bytes32 modelHash,
        uint256 numClients,
        uint256 timestamp
    );

    /**
     * @notice Emitted when ZKP proof hashes are anchored for a round.
     * @param round     FL training round number (1-based).
     * @param numProofs Number of accepted proof hashes.
     * @param timestamp Block timestamp of the anchor.
     */
    event ProofsAnchored(
        uint256 indexed round,
        uint256 numProofs,
        uint256 timestamp
    );

    // ── Storage ───────────────────────────────────────────────────────────────

    /// @notice The address that deployed this contract; the only address
    ///         permitted to write to it (the FL server coordinator).
    address public immutable coordinator;

    /// @notice Aggregated model hash per round.
    mapping(uint256 => bytes32) public modelHash;

    /// @notice Per-client update hashes per round.
    mapping(uint256 => bytes32[]) private _clientHashes;

    /// @notice Accepted ZKP proof hashes per round.
    mapping(uint256 => bytes32[]) private _proofHashes;

    /// @notice Client IDs (as bytes32 hashes) corresponding to each proof.
    mapping(uint256 => bytes32[]) private _proofClientIds;

    /// @notice Ordered list of rounds that have a model commit.
    uint256[] public committedRounds;

    /// @notice Ordered list of rounds that have proof anchors.
    uint256[] public anchoredRounds;

    // ── Constructor ───────────────────────────────────────────────────────────

    constructor() {
        coordinator = msg.sender;
    }

    // ── Access control ────────────────────────────────────────────────────────

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "FLLedger: caller is not coordinator");
        _;
    }

    // ── Write functions ───────────────────────────────────────────────────────

    /**
     * @notice Commit the aggregated model hash for a training round.
     *
     * @param round         FL round number (must not have been committed before).
     * @param _modelHash    SHA-256 of the aggregated plaintext model parameters,
     *                      encoded as a bytes32 value.
     * @param _clientHashes SHA-256 hashes of individual client updates included
     *                      in this round's aggregation.
     */
    function commitModel(
        uint256 round,
        bytes32 _modelHash,
        bytes32[] calldata _clientHashes
    ) external onlyCoordinator {
        require(
            modelHash[round] == bytes32(0),
            "FLLedger: round already committed"
        );
        modelHash[round] = _modelHash;
        _clientHashes[round] = _clientHashes;
        committedRounds.push(round);
        emit ModelCommitted(round, _modelHash, _clientHashes.length, block.timestamp);
    }

    /**
     * @notice Anchor ZKP proof hashes for a training round.
     *
     * @param round        FL round number.
     * @param _proofHashes SHA-256 hashes of accepted gnark Groth16 proof payloads.
     * @param _clientIds   SHA-256 hashes of the client identifiers corresponding
     *                     to each accepted proof (same length as _proofHashes).
     */
    function anchorProofs(
        uint256 round,
        bytes32[] calldata _proofHashes,
        bytes32[] calldata _clientIds
    ) external onlyCoordinator {
        require(
            _proofHashes.length == _clientIds.length,
            "FLLedger: proof and clientId arrays must have equal length"
        );
        _proofHashes[round] = _proofHashes;
        _proofClientIds[round] = _clientIds;
        anchoredRounds.push(round);
        emit ProofsAnchored(round, _proofHashes.length, block.timestamp);
    }

    // ── Read functions ────────────────────────────────────────────────────────

    /// @notice Return the aggregated model hash for *round*.
    function getModelHash(uint256 round) external view returns (bytes32) {
        return modelHash[round];
    }

    /// @notice Return the per-client update hashes for *round*.
    function getClientHashes(uint256 round) external view returns (bytes32[] memory) {
        return _clientHashes[round];
    }

    /// @notice Return the accepted ZKP proof hashes for *round*.
    function getProofHashes(uint256 round) external view returns (bytes32[] memory) {
        return _proofHashes[round];
    }

    /// @notice Return the client ID hashes corresponding to each proof in *round*.
    function getProofClientIds(uint256 round) external view returns (bytes32[] memory) {
        return _proofClientIds[round];
    }

    /// @notice Return the list of rounds that have a committed model hash.
    function getCommittedRounds() external view returns (uint256[] memory) {
        return committedRounds;
    }

    /// @notice Return the list of rounds that have anchored proof hashes.
    function getAnchoredRounds() external view returns (uint256[] memory) {
        return anchoredRounds;
    }

    /**
     * @notice Convenience view: return all audit data for a single round.
     * @return mHash       Aggregated model hash.
     * @return cHashes     Per-client update hashes.
     * @return pHashes     Accepted ZKP proof hashes.
     * @return pClientIds  Client ID hashes per proof.
     */
    function getRoundAudit(uint256 round)
        external
        view
        returns (
            bytes32 mHash,
            bytes32[] memory cHashes,
            bytes32[] memory pHashes,
            bytes32[] memory pClientIds
        )
    {
        mHash = modelHash[round];
        cHashes = _clientHashes[round];
        pHashes = _proofHashes[round];
        pClientIds = _proofClientIds[round];
    }
}
