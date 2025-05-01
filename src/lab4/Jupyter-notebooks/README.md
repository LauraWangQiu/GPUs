```bash
wsl
code .
# Create a virtual environment (venv)
python3 -m venv venv
source venv/bin/activate
# Install requirements (tensorflow, matplotlib, etc)
pip install notebook ipykernel tensorflow matplotlib
python -m ipykernel install --user --name=venv --display-name "venv"
# Select kernel in Jupyter Notebook and run kernels
```
