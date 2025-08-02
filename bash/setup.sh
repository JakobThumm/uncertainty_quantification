echo

# module load python3/3.12.4 cuda/12.4 cudnn/v8.9.7.29-prod-cuda-12.X 

# As aboved not working for local version: install cuda=12.4 and cudnn=9.10.2
python3 -m venv virtualenv
source virtualenv/bin/activate
python3 -m pip install --upgrade pip

# Update fix
# pip install --upgrade \
#   "jax[cuda12]" \
#   -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# python3 -m pip install --upgrade "jax[cuda12_pip]"==0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# python3 -m pip install --upgrade "jax[cuda12_pip]"==0.6.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install   "jax[cuda12-local]==0.4.34"   flax==0.10.4   orbax-checkpoint==0.11.5   -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# python3 -m pip install "jax[cuda12_pip]==0.5.1" \
#             jax-cuda12-plugin==0.5.3 \
#             jax-cuda12-pjrt==0.5.3 \
#             -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python3 -m pip install \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124



python3 -m pip install -r requirements_noversion.txt
python3 -m pip install   "jax[cuda12-local]==0.4.34"   flax==0.10.4   orbax-checkpoint==0.11.5   -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# python3 -m pip install --no-cache-dir --upgrade "jax[cuda12-local]"
# python3 -m pip install --no-cache-dir \
#   jax==0.6.0 \
#   jaxlib==0.6.0 \
#   jax-cuda12-plugin==0.6.0 \
#   -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python3 -m pip install --upgrade matfree==0.1.1
# python3 -m pip install --upgrade flax
# python -m pip install --upgrade orbax-checkpoint chex