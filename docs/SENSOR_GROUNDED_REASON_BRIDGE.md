# Provenance-aware conceptual verbalization after temporal aggregation

`tools/sensor_temporal_reason_bridge.py` answers a deliberately constrained
question: **which source-backed proposition can be stated in words from the
verified journal plus allowed positive ISA transitivity?** It does not assert
that any 3-component numerical vector is itself an English-language meaning.

To run:

```bash
python3 tools/sensor_temporal_reason_bridge.py snapshot.sensor.tsv.slices \
  --journal snapshot.sensor.tsv --subject penguin --relation ISA --object animal
```

The bridge reads independently checksum-verified MATS frames and calls the
existing strict journal/measurement consistency checker. It uses source
statements through the latest retained clock tick and computes:

- A **four-valued evidence state**: unknown / supports / refutes / both.
  Support and refutation are always separate source sets.
- A bounded 5-hop **positive ISA** proof path, with source names and event
  clocks. Unsupported transitive relations, negative ISA closure,
  negation-as-failure and contradictory explosion are forbidden.
- An explicitly journal-grounded phrase and readable sentence. A sentence
  describing conflicting evidence is not presented as a fact.

Native C++ `SensorMemory.reason` already implements the same restricted
positive-ISA inference. The new CI cross-check runs the actual native
`QUERY penguin ISA animal` against the Python independent readout, with
one supporting two-hop proof and a direct refutation. The secondary readout
is intended to reveal bugs, not install a second reasoner in production.

**Scientific separation:** The lexical identity originates in the journal,
with clocks aligned to the numeric sidecar but no cryptographic cross-file
atomicity. Native NLRP1 role probes (when enabled) learn to retrieve
previously seen *word identities* under new combinations; the graph bridge
then separately evaluates evidence and inference. Neither claims natural
language fluency, WordNet synset semantics, predictive memory benefit or
improvement in downstream cross-entropy. The native NaN16 root is unchanged.