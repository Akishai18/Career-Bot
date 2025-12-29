# 🤖 Career Bot

**Your AI-Powered Career Assistant**

Career Bot is an intelligent application that helps you excel in your career endeavors through AI-powered feedback on resumes, cover letters, and mock interviews with real-time computer vision analysis.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=flat&logo=google&logoColor=white)

---

## 🌟 What It Does

Career Bot is your all-in-one career preparation tool that provides:

- **AI-Powered Mock Interviews** with real-time feedback on your performance
- **Computer Vision Analysis** to track eye contact and engagement during interviews
- **Resume Optimization** with intelligent suggestions for improvement
- **Cover Letter Enhancement** with personalized feedback
- **Speech Recognition** to analyze your spoken responses
- **Text-to-Speech** for natural conversational practice

Whether you're preparing for your dream job or polishing your application materials, Career Bot provides the feedback and practice you need to succeed.

---

## ✨ Key Features

### 🎥 Smart Interview Practice
- **Real-time eye tracking** using Haar Cascade classifiers
- **Face detection** to monitor engagement and body language
- **Live feedback** on whether you're maintaining eye contact with the camera
- **Visual indicators** showing your eye tracking status during practice

### 🎤 Speech Analysis
- **Speech-to-text conversion** using Google Speech Recognition
- **Intelligent feedback** on your interview answers
- **Analysis of delivery skills** including volume, pitch, melody, and articulation
- **Practice with custom questions** tailored to your target job and question type

### 📝 Resume & Cover Letter Support
- **AI-powered feedback** from Google Gemini
- **Personalized suggestions** for improving your application materials
- **Conversational interface** for iterative refinement

### 🔧 Technical Implementation
- **Computer Vision:** OpenCV with Haar Cascade for face and eye detection
- **AI Integration:** Google Gemini API for intelligent feedback generation
- **Multi-threading:** Simultaneous camera feed and speech recognition
- **Audio Processing:** pyttsx3 for text-to-speech and speech_recognition for STT

---

## 🏗️ How It Works

### Eye Contact Detection Algorithm

Career Bot uses advanced computer vision techniques to determine if you're maintaining eye contact:

1. **Face Detection:** Uses Haar Cascade classifiers to detect faces in real-time
2. **Eye Detection:** Identifies both eyes within the detected face region
3. **Center Calculation:** Computes the center coordinates of each eye
4. **Angle Analysis:** Calculates the angle between the two eye centers
5. **Gaze Direction:** Determines if you're looking at the camera based on angular threshold (±15°)

```python
# Simplified algorithm
angle = calculate_angle(left_eye_center, right_eye_center)
if abs(angle) < 15:  # Looking straight at camera
    display_status = "Looking at Camera"
```

### Interview Workflow

1. **Question Generation:** Specify your job type and question category (behavioral, technical, practical)
2. **AI Question:** Google Gemini generates a relevant interview question
3. **Parallel Processing:** Camera feed and speech recognition start simultaneously using threading
4. **Real-time Monitoring:** Visual feedback on eye contact while you speak
5. **Answer Analysis:** Your spoken response is transcribed and analyzed
6. **Comprehensive Feedback:** AI provides detailed feedback on content and delivery

---

## 🧩 Tech Stack

**Computer Vision:**
- OpenCV
- Haar Cascade Classifiers (`haarcascade_frontalface_default.xml`, `haarcascade_eye_tree_eyeglasses.xml`)

**AI & NLP:**
- Google Gemini API (gemini-1.5-pro-latest)
- Speech Recognition (Google Speech Recognition)
- pyttsx3 (Text-to-Speech)

**Core:**
- Python 3.x
- Threading for concurrent operations
- NumPy for mathematical operations

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Webcam
- Microphone
- Google Gemini API key

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/career-bot.git
   cd career-bot
   ```

2. **Install required packages:**
   ```bash
   pip install opencv-python
   pip install numpy
   pip install google-generativeai
   pip install SpeechRecognition
   pip install pyttsx3
   pip install pyaudio
   ```

3. **Download Haar Cascade files:**
   - Download `haarcascade_frontalface_default.xml`
   - Download `haarcascade_eye_tree_eyeglasses.xml`
   - Place both files in the same directory as the main script
   - Available from: [OpenCV GitHub Repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)

4. **Set up Google Gemini API:**
   - Obtain your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace the API key in the code:
     ```python
     genai.configure(api_key="YOUR_API_KEY_HERE")
     ```

5. **Run the application:**
   ```bash
   python career_bot.py
   ```

---

## 🎯 Usage Guide

### Starting Career Bot

```bash
python career_bot.py
```

You'll be greeted with:
```
Hello Welcome to Career Bot
```

### Main Menu Options

1. **Resume** - Get feedback on your resume
2. **Cover Letter** - Improve your cover letter
3. **Interview** - Practice mock interviews with AI
4. **Exit** - Close the application

### Interview Practice

1. Enter your target job type (e.g., "Software Engineer")
2. Choose question type:
   - **Behavioral:** Situational questions about past experiences
   - **Technical:** Job-specific technical knowledge
   - **Practical:** Problem-solving and scenario-based questions
3. Read the AI-generated question
4. Type `ready` when prepared to answer
5. The camera feed opens and starts tracking your eye contact
6. Speak your answer clearly into the microphone
7. Receive comprehensive feedback on both content and delivery
8. Press `q` to close the camera when done

### Resume & Cover Letter Feedback

1. Select your option from the main menu
2. Describe what you need help with
3. Receive AI-powered suggestions for improvement
4. Iterate and refine based on feedback

---

## 📊 What Gets Analyzed

### During Interviews:

**Visual Feedback:**
- Eye contact consistency
- Face detection and positioning
- Real-time engagement indicators

**Speech Analysis:**
- Answer content and relevance
- Communication clarity
- Volume and projection
- Pitch and melody
- Articulation

**Overall Assessment:**
- Interview skills rating
- Specific improvement suggestions
- Strengths and areas for growth

---

## 🔧 Technical Details

### Computer Vision Pipeline

```python
1. Capture frame from webcam
2. Convert to grayscale for processing
3. Detect faces using Haar Cascade
4. For each face:
   - Extract region of interest (ROI)
   - Detect eyes within face region
   - Calculate eye center coordinates
   - Compute angle between eyes
   - Determine gaze direction
5. Display visual feedback on frame
```

### Threading Architecture

Career Bot uses multi-threading to handle concurrent operations:
- **Camera Thread:** Continuously captures and processes video frames
- **Speech Thread:** Listens for and transcribes audio input
- **Main Thread:** Coordinates UI and AI interactions

---

## ⚙️ Configuration

### Eye Contact Sensitivity

Adjust the angle threshold for eye contact detection:
```python
angle_limit = 15  # degrees (default)
# Lower value = stricter eye contact requirement
# Higher value = more lenient detection
```

### AI Model Settings

Customize the Gemini model parameters:
```python
generation_config = {
    "temperature": 1,        # Creativity (0-2)
    "top_p": 0.95,          # Nucleus sampling
    "top_k": 0,             # Top-k sampling
    "max_output_tokens": 8192  # Response length
}
```

---

## 🚀 Future Enhancements

- **Body language analysis:** Posture and gesture tracking
- **Emotion detection:** Facial expression analysis during interviews
- **Session recording:** Save and review past practice sessions
- **Progress tracking:** Monitor improvement over time
- **Custom question banks:** Build your own interview question database
- **Multi-language support:** Practice in different languages
- **Web interface:** Browser-based UI for easier access
- **Mobile app:** Practice interviews on the go

---

## 🐛 Troubleshooting

### Common Issues

**Camera not opening:**
- Ensure your webcam is connected and not in use by another application
- Check camera permissions in your system settings

**Speech recognition not working:**
- Verify your microphone is properly connected
- Check microphone permissions
- Ensure you have an active internet connection (Google Speech Recognition requires it)

**API errors:**
- Verify your Google Gemini API key is valid
- Check your API usage limits
- Ensure you have an active internet connection

**Haar Cascade files not found:**
- Confirm the XML files are in the same directory as the script
- Check file names match exactly (case-sensitive)

---

## 📋 Requirements

```
opencv-python>=4.5.0
numpy>=1.19.0
google-generativeai>=0.3.0
SpeechRecognition>=3.8.0
pyttsx3>=2.90
PyAudio>=0.2.11
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve Career Bot:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 💡 Tips for Best Results

**For Eye Contact Detection:**
- Sit directly facing the camera
- Ensure good lighting on your face
- Maintain a consistent distance from the camera
- Avoid excessive head movement

**For Speech Recognition:**
- Speak clearly and at a moderate pace
- Use a quality microphone if possible
- Minimize background noise
- Ensure stable internet connection

---
