# Create local environment

python -m venv venv

# Activate local environment

venv\Scripts\activate

# Deactivate local environment

deactivate

# Install package

pip install numpy
pip install -r requirements.txt

# Export all package

pip freeze > requirements.txt

# Docker command

docker build -t service_calculations .
