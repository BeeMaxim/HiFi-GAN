# Implementation of HiFi-gan

<p align="center">
  <a href="#instruction">Instruction</a> 
  <a href="#weights">Weights</a>•
  <a href="#license">License</a>
</p>

## Instruction

To synthesize from audio use this command:
```python resynthesize.py input_path=PATH_TO_GT_FOLDER output_path=PATH_TO_OUTPUT_FOLDER from_pretrained=PATH_TO_CHECKPOINT```

To synthesize from text use this command:
```python synthesize.py input_path=PATH_TO_GT_FOLDER output_path=PATH_TO_OUTPUT_FOLDER from_pretrained=PATH_TO_CHECKPOINT```

or 

```python synthesize.py output_path=PATH_TO_OUTPUT_FOLDER text=TEXT file_name=NAME_OF_OUTPUT_FILE from_pretrained=PATH_TO_CHECKPOINT```

to synthesize one audio from TEXT

## Weights
https://drive.google.com/file/d/1iGW68Afr9MrP9Yh5wKuRzL3N9JID0x9F/view?usp=sharing

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
