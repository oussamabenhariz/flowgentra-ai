# Contributing to FlowgentraAI

Thank you for your interest in contributing to FlowgentraAI! We welcome contributions of all kinds.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/FlowgentraAI.git
   cd FlowgentraAI
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development

### Prerequisites
- Rust 1.70+
- Cargo

### Build the Project
```bash
cargo build
```

### Run Tests
```bash
cargo test
```

### Check Code Quality
```bash
cargo check
cargo fmt
cargo clippy
```

## Making Changes

1. Make your changes in your feature branch
2. Ensure all tests pass: `cargo test`
3. Format your code: `cargo fmt`
4. Check for linting issues: `cargo clippy`

## Submitting Changes

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear description of what you changed and why
   - Reference to any related issues
   - Tests for new functionality

3. **Address feedback** from code review

## Project Structure

```
flowgentra-ai/              - Main library crate
├── src/
│   ├── core/            - Core AI agent framework
│   ├── lib.rs
│   └── main.rs
└── Cargo.toml

flowgentra-ai-macros/       - Procedural macros crate
├── src/
└── Cargo.toml
```

## Documentation

- See `flowgentra-ai/README.md` for feature overview
- See `flowgentra-ai/QUICKSTART.md` for quick start guide
- See `flowgentra-ai/DEVELOPER_GUIDE.md` for architecture details
- See `flowgentra-ai/CONFIG_GUIDE.md` for configuration reference

## Code Style

We follow standard Rust conventions:
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Write descriptive variable and function names
- Add documentation comments for public APIs

## Questions?

Feel free to open an issue or discussion for questions about contributing.

Thank you for helping make FlowgentraAI better! 🎉
