# Automated Protocol Run

This example demonstrates a programmatic way to:
1. Load the scope API
1. Set required basic configuration for the scope API (objective, wellplate, offsets, etc)
1. Load an existing protocol file to an instance of the `Protocol` class
1. Pass the protocol instance to an instance of the `SequencedCaptureExecutor`
1. Command the sequenced capture executor instance to run the protocol
