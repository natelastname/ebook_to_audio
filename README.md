# eBook to Audio Converter

Welcome to the eBook to Audio Converter project! This tool transforms your eBooks into high-quality audio files, making it easier to enjoy your favorite books on the go.

## Project Description

The eBook to Audio Converter is designed to convert various eBook formats into audio files using advanced text-to-speech (TTS) technology. Whether you're commuting, exercising, or simply relaxing, this tool allows you to listen to your eBooks anytime, anywhere.

## Features

- **Multi-format Support**: Convert eBooks from formats like EPUB and PDF.
- **Bilingual Conversion**: Supports text-to-speech in multiple languages.
- **Customizable Output**: Choose from different voices and audio settings.
- **Efficient Processing**: Fast conversion with minimal resource usage.
- **User-friendly CLI**: Easy-to-use command-line interface for quick operations.

## Technology Stack

- **Python**: Core programming language.
- **Click**: For building command-line interfaces.
- **Pydantic**: Data validation and settings management.
- **TTS Engines**: Integration with various TTS engines for diverse voice options.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd ebook-to-audio
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Run the Application**:
   ```bash
   poetry run e2a_cli
   ```

## Sample Usage

Convert an EPUB file to audio:
```bash
poetry run e2a_cli do_tts --input your_ebook.epub --output output_audio.mp3
```

## Extending the Project

This project can be extended to support additional eBook formats, integrate with more TTS engines, or even provide a graphical user interface for non-technical users. It can also be adapted for educational purposes, such as language learning tools or accessibility applications for visually impaired users.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the GPLv3 license. See the LICENSE file for more details.

## Contact

For questions or feedback, please contact the author at nate@example.com.
