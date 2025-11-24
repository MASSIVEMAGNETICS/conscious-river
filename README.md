# conscious-river

All input streams flow into a river - A complete consciousness engine with multiple sensory inputs, attention mechanisms, memory, and intelligent data merging.

## Features

✨ **Multiple Sensory Streams** - Visual, auditory, emotional, cognitive, and more  
🎯 **Attention Mechanism** - Dynamic attention weighting with softmax-based prioritization  
🧠 **Memory System** - Importance-weighted memory with automatic salience-based pruning  
🔀 **Stream Merging** - Intelligent fusion of all streams into unified consciousness  
📊 **Data Pruning** - Automatic removal of insignificant data to maintain efficiency  
⚙️ **System Instructions** - Configurable behavioral guidelines integrated into processing  

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the consciousness engine demo
python consciousness_engine.py

# Run the full integration example
python integration_example.py
```

## Documentation

- **[Consciousness Engine Documentation](CONSCIOUSNESS_ENGINE.md)** - Complete API and architecture guide
- **[Integration Example](integration_example.py)** - Full cognitive agent simulation
- **[Original River System](river.py)** - Extended holonic architecture with GUI

## Example Usage

```python
from consciousness_engine import ConsciousnessEngine

# Define your streams
streams = ["visual", "auditory", "emotional", "cognitive"]

# Create engine
engine = ConsciousnessEngine(
    stream_names=streams,
    memory_capacity=500,
    system_instructions="Integrate sensory data coherently"
)

# Ingest data
engine.ingest_stream("visual", {"scene": "forest"}, importance=0.8)
engine.ingest_stream("emotional", {"mood": "calm"}, importance=0.6)

# Process consciousness
state = engine.process_consciousness()
print(engine.get_consciousness_summary())
```

## Architecture

The consciousness engine integrates:

1. **Multiple Sensory Inputs** - Configurable streams for any type of data
2. **Attention Mechanism** - Weighted prioritization with smooth transitions
3. **Memory with Importance** - Salience-based storage and retrieval
4. **Automatic Pruning** - Removes low-salience data to maintain capacity
5. **Stream Merger** - Combines all streams into coherent consciousness state
6. **System Instructions** - Behavioral guidelines integrated throughout

See [CONSCIOUSNESS_ENGINE.md](CONSCIOUSNESS_ENGINE.md) for detailed architecture documentation.

## Requirements

- Python 3.8+
- numpy >= 1.24.0
- matplotlib >= 3.7.0 (optional)

## License

See repository license.
