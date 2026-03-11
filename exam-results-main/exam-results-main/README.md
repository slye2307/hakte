
 # Exam Prediction App

An intelligent Flask web application that predicts student exam results based on lifestyle and study habits using machine learning.

## Features
- 🎯 Predicts exam scores using Random Forest Classifier (92% accuracy)
- 📊 Personalized improvement suggestions based on student data
- 📄 PDF report generation with detailed results
- 📱 Responsive web interface
- 🚀 Fast ML predictions with joblib model persistence

## Tech Stack
- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Model**: Random Forest Classifier
- **PDF Generation**: ReportLab

## Quick Start

### Local Development
```bash
# Clone and setup
git clone <your-repo>
cd exam-results-main
pip install -r requirements.txt

# Run the app
cd apn
python app.py
```

### Deployment
This app is ready for deployment on:
- **PythonAnywhere** (easiest for beginners)
- **Render** (recommended - always-on free tier)
- **Vercel** (modern serverless)
- **Heroku** (classic choice)

## Model Performance
- **Accuracy**: 92.36%
- **Dataset**: 14,003 student records
- **Features**: 9 input parameters
- **Classes**: Excellent, Good, Satisfactory, Fail

## API Endpoints
- `GET /` - Home page with input form
- `POST /predict` - Generate prediction and suggestions
- `GET /download_result` - Download PDF report

## License
MIT License - feel free to use and modify!
