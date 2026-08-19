

<p align="center">
  <img src="Assets/banner.png" alt="Banner" width="100%">
</p>


EchoHands is an AI-powered sign language recognition framework designed to recognize hand signs through camera input and convert them into understandable digital output.

The project follows a modular architecture covering the complete recognition pipeline, including data collection and management, hand and landmark processing, model training, evaluation, testing, real-time recognition, sequence detection, and word building.

The current implementation is trained and tested primarily on American Sign Language (ASL). The framework is designed in a modular manner, allowing it to be extended and adapted to other sign languages using suitable datasets in the future.

---

## Supported Signs

EchoHands currently works with the trained ASL signs included in the project. Static signs are handled through the recognition pipeline, while dynamic signs such as **J** and **Z** are handled using sequence-based processing.



<p align="center">
  <img src="sign description/sign letters.png" alt="Banner" width="750">
</p>

<p align="center">
  Figure 1. ASL alphabet and numeric signs supported by the EchoHands framework
</p>



---

## Installation
1. Clone the repository

Clone the EchoHands repository to your local machine.

```bash
git clone https://github.com/shivanshu43/EchoHands
```
2. Requirements

To run EchoHands, the required Python dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The project requires:

- Python
- Camera access for real-time recognition
- Required computer vision and machine learning dependencies listed in `requirements.txt`
- 
3. Create a virtual environment
```bash
python -m venv venv
```
4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```


## Usage

EchoHands provides separate modules for different stages of the sign language recognition workflow.

The project can be used for:

1. Collecting and managing sign language data.
2. Checking dataset quality and variation.
3. Generating and processing landmark-based features.
4. Training recognition models.
5. Evaluating and validating trained models.
6. Performing static sign recognition.
7. Detecting dynamic sign sequences.
8. Performing real-time camera-based recognition.
9. Building recognized signs into words.

Run the appropriate script depending on the module you want to use.

For example:

```python test.py```

Additional modules are available inside the src directory for dataset management, training, recognition, testing, and analysis.

## How It Works

EchoHands follows a modular AI-powered recognition pipeline.

The general workflow is:

**Camera Input → Hand Detection → Landmark Processing → Feature Processing → AI Model → Sign Recognition → Sequence Detection → Word Building → Digital Output**

During real-time recognition, the camera captures the hand gesture. The system processes the detected hand and its landmarks before passing the relevant information to the trained recognition model.

For static signs, the trained recognition pipeline predicts the corresponding sign from the captured input.

For dynamic signs such as J and Z, EchoHands uses sequence-based processing to analyze the movement across multiple frames before generating a prediction.

Recognized signs can then be passed through sequence and word-building components to support higher-level sign interpretation.

[INSERT PROJECT WORKFLOW / ARCHITECTURE IMAGE HERE]
Figure 2. High-level working of the EchoHands recognition framework.


## Project Structure
```bash
EchoHands/
│
├── assets/
├── data/
├── models/
├── sign description/
├── src/
├── tests/
├── test.py
├── requirements.txt
├── README.md
```
## File and Module Descriptions

`data/`
Contains project data used for sign recognition, processing, and training.

The project includes processed sequence data used for dynamic sign recognition, including samples for signs such as J and Z with left- and right-hand variations.


`models/`
Contains trained model artifacts and related model files.

The project includes model resources such as:

- Static recognition model files
- Dynamic sequence model files
- Random Forest model artifacts
- Label encoders
- Model metadata


`src/`
Contains the main source code for the EchoHands framework.

The source code is organized into modules covering the major stages of the system.

Core Recognition Modules

The core modules handle components such as:

- Camera processing
- Hand detection
- Landmark processing
- Sign prediction
- Recognition control
- Sequence detection
- Word building

These modules form the main pipeline used for camera-based sign recognition.

## Dataset Modules

The dataset-related modules support:

- Data collection
- Duplicate detection
- Dataset quality checking
- Dataset variation management
- Sequence generation
- Dataset analysis
- Data export
- Dataset visualization and dashboard functionality

These components support the preparation and management of data used by the recognition framework.

## Training Modules

The training-related modules support:

- Dataset generation
- Geometric feature processing
- Data augmentation
- Model training
- Model evaluation
- Validation
- Cross-validation
- Error analysis

These modules are responsible for developing and evaluating the recognition models.

`requirements.txt`
Contains the Python dependencies required to run the EchoHands project.

## Dynamic Sign Recognition

Unlike static signs that can be recognized from an individual frame, certain ASL signs involve movement.

EchoHands includes a dedicated sequence-based approach for dynamic signs.

The dynamic recognition workflow analyzes hand information across a sequence of frames rather than relying only on a single image.

This enables the framework to handle signs such as:

- J
- Z

The project includes processed sequence data and a dynamic model for this part of the recognition pipeline.

[INSERT DYNAMIC SIGN / SEQUENCE RECOGNITION IMAGE HERE]

Figure 3. Sequence-based recognition used for dynamic ASL signs.

**Examples**

The following examples demonstrate the EchoHands recognition process.

[INSERT STATIC SIGN RECOGNITION SCREENSHOT HERE]
Example 1. Real-time recognition of a static ASL sign.

[INSERT DYNAMIC SIGN RECOGNITION SCREENSHOT HERE]
Example 2. Sequence-based recognition of a dynamic ASL sign.

[INSERT CAMERA RECOGNITION OUTPUT SCREENSHOT HERE]
Example 3. Camera-based sign recognition and digital output.

## Current Implementation

The current version of EchoHands includes a modular framework for:

- Sign language data collection and management
- Dataset quality and variation handling
- Landmark and feature processing
- Static sign recognition
- Dynamic sequence recognition
- Model training
- Model evaluation and validation
- Error analysis
- Camera-based recognition
- Sequence detection
- Word building
- Testing and experimentation

The current trained implementation is focused primarily on:

**American Sign Language (ASL)**

EchoHands is currently maintained as a project repository and modular codebase.

## Future Improvements

The following improvements are planned for the future development of EchoHands:

- Adapt the framework to additional sign languages using suitable datasets.
- Expand the supported sign vocabulary and datasets.
- Further improve static and dynamic sign recognition performance.
- Improve recognition robustness under different lighting and camera conditions.
- Enhance sequence-based recognition for additional motion-based signs.
- Improve word and higher-level sign interpretation capabilities.
- Develop a mobile-compatible camera interface so that a smartphone camera can potentially be used as a portable recognition device.
- Explore deployment of the trained recognition framework as a cloud-based service.
- Enable future access to the recognition system from compatible camera-enabled devices.
- Improve the user interface and overall accessibility of the platform.

[!NOTE]Note: Mobile camera integration, cloud deployment, multi-device accessibility, and adaptation to additional sign languages are future development plans and are not part of the current deployed implementation.

## Future Vision

The long-term vision of EchoHands is to evolve from a modular ASL recognition framework into a more accessible and extensible sign language recognition platform.

The planned direction of development is:

**Current ASL Framework → Additional Sign Languages → Mobile Camera Integration → Cloud-Based Recognition → Multi-Device Accessibility**

The modular architecture of EchoHands provides a foundation for extending the system while keeping the current implementation focused on its core sign recognition workflow.
