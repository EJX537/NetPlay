#!/bin/bash

# Install litellm with tokenizers workaround for Python 3.8

echo "Installing tokenizers < 0.20 with --no-build-isolation..."
pip install --no-build-isolation 'tokenizers<0.20'

echo ""
echo "Installing compatible versions for Python 3.8..."
pip install 'openai>=1.0,<2.0' 'litellm==1.78.7'

echo ""
echo "Verifying installation..."
python -c "import litellm; print(f'✓ litellm {litellm.__version__} installed successfully')"
