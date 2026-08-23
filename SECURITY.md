# Security policy

QuantBenchie is intended for evaluating local language models. It does not provide a sandbox for executing arbitrary model-generated code or tool calls.

Do not run generated code, shell commands, plugins, or model downloads in a privileged environment. Use an isolated worker for code-execution benchmarks and treat model output as untrusted input.

To report a security issue in the harness, open a private report with the maintainers before publishing exploit details. Do not include access tokens, private model URLs, or personal data in an issue.
