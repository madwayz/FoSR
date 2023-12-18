## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/madwayz/FoSR.git
```

Navigate to the project root directory:

```bash
cd jupyter-docker-compose
```

Build the the image for the Jupyter Notebook server:

```bash
docker-compose build
```

Start the Jupyter Notebook server:

```bash
docker-compose up
```

After running this command, the Jupyter Notebook server should be accessible at `http://localhost:8888`.


## Directory Structure

- `./work/notebooks`: This is the directory where you can add your Jupyter notebooks. It's mounted as a volume in the Docker container, so notebooks created and saved in the Jupyter Notebook IDE will persist here.
- `./workspace` - This is the directory where you can your python libraries.

## Notes

The `requirements.txt` file is copied to the Docker container during the build process, and the Python dependencies listed within are installed. To add or update dependencies, modify this file, then rebuild the Docker image.

