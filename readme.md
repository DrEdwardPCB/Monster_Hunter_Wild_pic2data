# Monster hunter wild Pic2data

This repo is created to help the character creation process of monster hunter wild. this code leverages open CV and dlib to detect face inside image and paramertize to monster hunter wild type.

## Disclaimer

Since I personally does not allow to purchase or open Monster hunter wild, I cannot verify the code itself
The code is mainly generated using Claude plus minor fixes on lib dependencies and logging, use at your own risk

# How to use
## Setup
1. clone this repo
2. install poetry and python
3. Setup virtual env by `poetry use env 3.12`
4. install dependencies by `poetry install`
5. download the model data for `dlib` by `wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` then `bunzip2 shape_predictor_68_face_landmarks.dat.bz2`
6. locate your image in `.jpg` format that wants to be process
7. run the following command `poetry run python mhw_character_generator.py`

## output
the output will be the in json format contains in the file `{{filename}}_mhw_params.json`
## Sample generate from my photo
```json
{
  "face_type": 1,
  "skin_color": 5,
  "skin_color_rgb": [
    82,
    58,
    45
  ],
  "face_width": 63,
  "jaw_width": 48,
  "chin_height": 29,
  "eye_type": 2,
  "eye_position_vertical": 50,
  "eye_position_horizontal": 27,
  "eye_size": 100,
  "eye_color": 0,
  "eye_color_rgb": [
    61,
    48,
    40
  ],
  "nose_type": 9,
  "nose_height": 50,
  "nose_width": 21,
  "nose_length": 27,
  "mouth_type": 10,
  "mouth_width": 41,
  "mouth_height": 100,
  "upper_lip_thickness": 100,
  "lower_lip_thickness": 100,
  "hair_color": 1,
  "hair_color_rgb": [
    59,
    51,
    50
  ],
  "hair_style": 0,
  "eyebrow_type": 0,
  "makeup": 0,
  "facial_hair": 0,
  "scars": 0,
  "age": 0
}
```

## final words
hope this tools help and let me know if I can improve it. Happy Hunting