# Reinforcement Learning Study Notes

A comprehensive, bilingual (English/中文) study note system for Reinforcement Learning that is interview-ready, mathematically rigorous, and continuously extensible.

## Features

- **Bilingual Content**: Full English and Chinese documentation with parallel structure
- **Interview-Ready**: Each topic includes interview summaries and "what to memorize" sections
- **Mathematical Rigor**: Key equations with clear derivations and explanations
- **Interactive Quizzes**: 5+ questions per topic with click-to-reveal answers
- **Runnable Code Examples**: Minimal but complete algorithm implementations
- **Q&A System**: Local search and patch proposal workflow

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Serve Documentation Locally

```bash
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Run Algorithm Examples

```bash
# List available algorithms
python -m rl_examples.run --list

# Run a specific algorithm
python -m rl_examples.run --algo q_learning
python -m rl_examples.run --algo dqn
python -m rl_examples.run --algo ppo
```

### Use the Q&A System

```bash
# Ask a question
python -m qa.ask "What is the Bellman equation?"

# Ask with patch proposal
python -m qa.ask "I don't understand TD learning" --propose-patch

# Build the search index
python -m qa.index --rebuild
```

### Validate Documentation

```bash
python -m qa.validate
```

## Topics Covered

### Fundamentals
- MDP Basics (states, actions, rewards, transitions)
- Policy and Value Functions (V, Q, advantage)
- Bellman Equations (expectation and optimality)

### Dynamic Programming
- Policy Evaluation
- Policy Iteration
- Value Iteration

### Model-Free Methods
- Monte Carlo Methods
- SARSA (on-policy TD)
- Q-Learning (off-policy TD)
- Expected SARSA

### Function Approximation
- Linear Methods
- Neural Network Approximation

### Deep Reinforcement Learning
- DQN (with replay buffer, target network)
- Policy Gradients (REINFORCE)
- Actor-Critic Methods
- PPO (Proximal Policy Optimization)

### Advanced Topics
- Exploration Strategies
- Generalized Advantage Estimation (GAE)
- Stability Issues (deadly triad)
- Practical Training Pipeline

## Project Structure

```
.
├── docs/
│   ├── en/                 # English documentation
│   │   ├── fundamentals/   # MDP, policy-value, bellman
│   │   ├── dp/             # Dynamic programming
│   │   ├── mc/             # Monte Carlo methods
│   │   ├── td/             # Temporal difference
│   │   ├── fa/             # Function approximation
│   │   ├── deep/           # Deep RL (DQN, PG, PPO)
│   │   ├── advanced/       # Exploration, GAE, stability
│   │   └── interview/      # Common interview questions
│   ├── zh/                 # Chinese documentation (mirrored)
│   └── assets/             # Shared assets (JS, images)
├── rl_examples/
│   ├── algorithms/         # Algorithm implementations
│   │   ├── policy_iteration.py
│   │   ├── value_iteration.py
│   │   ├── mc_control.py
│   │   ├── sarsa.py
│   │   ├── q_learning.py
│   │   ├── expected_sarsa.py
│   │   ├── dqn.py
│   │   ├── reinforce.py
│   │   └── ppo.py
│   └── run.py              # CLI entry point
├── qa/
│   ├── ask.py              # Q&A tool
│   ├── index.py            # Document indexer
│   ├── patch.py            # Patch management
│   └── validate.py         # Documentation validator
├── proposals/              # Content update proposals
├── skills/                 # Reusable skill templates
│   ├── template.md
│   ├── bilingual-knowledge-base.md
│   └── bilingual-technical-notes.md
├── .claude/                # Temporary files (git-ignored)
├── mkdocs.yml              # MkDocs configuration
├── requirements.txt        # Python dependencies
├── DECISIONS.md            # Design decisions log
└── README.md               # This file
```

## Extending the Knowledge Base

### Adding a New Topic

1. **Create English page**: `docs/en/[section]/[topic].md`
2. **Create Chinese page**: `docs/zh/[section]/[topic].md` (same structure)
3. **Add to navigation**: Update `mkdocs.yml` nav section
4. **Validate**: Run `python -m qa.validate`

### Required Sections per Page

Each topic page must include these sections in order:

1. **Interview Summary** (3-6 lines, include "What to memorize")
2. **Core Definitions** (formal definitions with notation)
3. **Math and Derivations** (key equations with explanations)
4. **Algorithm Sketch** (pseudocode, complexity)
5. **Common Pitfalls** (3-5 numbered items)
6. **Mini Example** (small worked example)
7. **Quiz** (5+ questions using `<details>` tags)
8. **References** (books, papers, "what to memorize")

### Quiz Question Format

```html
<details>
<summary><strong>Q1 (Conceptual):</strong> Question text here?</summary>

**Answer**: Direct answer in 1-2 sentences.

**Explanation**: Why this is correct (2-4 sentences).

**Key equation**: $$ equation $$

**Common pitfall**: What mistake people make.
</details>
```

Each quiz must have:
- 2 conceptual questions
- 2 math/derivation questions
- 1 practical/debugging question

### Adding a New Algorithm

1. **Create implementation**: `rl_examples/algorithms/[algo].py`
2. **Include docstring**: Link to corresponding docs page
3. **Add main()**: Entry point for CLI
4. **Register in run.py**: Add to ALGORITHMS dict
5. **Link from docs**: Add code link in the topic page

## Configuration

### MkDocs

Site configuration is in `mkdocs.yml`:
- Theme: Material
- Language: EN (default), ZH via i18n plugin
- Math: MathJax
- Search: Built-in per-language

### Environment Variables (Optional)

```bash
# Enable LLM integration for Q&A (disabled by default)
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

## Development

### Build Site

```bash
mkdocs build  # Output to site/
```

### Run Tests

```bash
# Validate documentation structure
python -m qa.validate

# Run algorithm tests (requires gymnasium)
python -m rl_examples.run --algo value_iteration
```

### Rebuild Search Index

```bash
python -m qa.index --rebuild
```

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for rationale behind:
- MkDocs Material over Docusaurus
- MathJax over KaTeX
- PyTorch for deep RL examples
- BM25 for offline Q&A search

## Contributing

1. Follow the section structure template
2. Ensure EN/ZH pages are mirrored
3. Include 5+ quiz questions per topic
4. Test code examples before committing
5. Run validation before submitting

## License

MIT License

## References

- **Sutton & Barto**, Reinforcement Learning: An Introduction (2nd ed.)
- **David Silver's RL Course**, DeepMind
- **Spinning Up in Deep RL**, OpenAI
- **CS 285**, UC Berkeley Deep RL Course
