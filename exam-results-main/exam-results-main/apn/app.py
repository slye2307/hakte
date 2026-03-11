from flask import Flask, request, render_template, send_file, session
import joblib
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime

app = Flask(__name__)
app.secret_key = 'exam-results-secret-key-2026'

# Mapping for converting numeric form values to readable text
VALUE_MAPPINGS = {
    'Resources': {
        '0': 'No Resources',
        '1': 'Some Resources',
        '2': 'Many Resources'
    },
    'Internet': {
        '0': 'No Internet',
        '1': 'Yes'
    },
    'Gender': {
        '0': 'Female',
        '1': 'Male'
    },
    'StressLevel': {
        '0': 'Low',
        '1': 'Medium',
        '2': 'High'
    }
}

def convert_details_to_readable(details):
    """Convert numeric form values to human-readable text"""
    readable_details = {}
    for key, value in details.items():
        if key in VALUE_MAPPINGS:
            readable_details[key] = VALUE_MAPPINGS[key].get(str(value), str(value))
        else:
            readable_details[key] = str(value)
    return readable_details

def generate_improvement_suggestions(details, prediction_result):
    """Generate personalized improvement suggestions based on student data"""
    suggestions = []
    
    # Convert to appropriate types for comparison
    study_hours = float(details.get('StudyHours', 0))
    attendance = float(details.get('Attendance', 0))
    assignment_completion = float(details.get('AssignmentCompletion', 0))
    online_courses = float(details.get('OnlineCourses', 0))
    stress_level = details.get('StressLevel', '0')
    resources = details.get('Resources', '0')
    internet = details.get('Internet', '0')
    
    # Study Hours suggestions
    if study_hours < 15:
        suggestions.append({
            'priority': 'high',
            'category': 'Study Hours',
            'current': f"{study_hours} hours/week",
            'recommendation': 'Increase study hours to at least 20-25 hours per week for better results',
            'action': 'Create a study schedule and dedicate consistent time daily'
        })
    elif study_hours < 20:
        suggestions.append({
            'priority': 'medium',
            'category': 'Study Hours',
            'current': f"{study_hours} hours/week",
            'recommendation': 'Consider increasing study hours to 20-25 hours per week',
            'action': 'Focus on quality study time with minimal distractions'
        })
    
    # Attendance suggestions
    if attendance < 75:
        suggestions.append({
            'priority': 'high',
            'category': 'Attendance',
            'current': f"{attendance}%",
            'recommendation': 'Improve attendance to above 80% for better academic performance',
            'action': 'Attend all classes and participate actively in discussions'
        })
    elif attendance < 85:
        suggestions.append({
            'priority': 'medium',
            'category': 'Attendance',
            'current': f"{attendance}%",
            'recommendation': 'Try to maintain attendance above 85%',
            'action': 'Set reminders for classes and prioritize important sessions'
        })
    
    # Assignment Completion suggestions
    if assignment_completion < 70:
        suggestions.append({
            'priority': 'high',
            'category': 'Assignments',
            'current': f"{assignment_completion}%",
            'recommendation': 'Complete at least 80% of assignments to improve understanding',
            'action': 'Start assignments early and seek help when needed'
        })
    elif assignment_completion < 85:
        suggestions.append({
            'priority': 'medium',
            'category': 'Assignments',
            'current': f"{assignment_completion}%",
            'recommendation': 'Aim for 90%+ assignment completion',
            'action': 'Create a timeline for assignment submissions'
        })
    
    # Online Courses suggestions
    if online_courses < 5:
        suggestions.append({
            'priority': 'low',
            'category': 'Online Learning',
            'current': f"{online_courses} courses",
            'recommendation': 'Consider taking 5-10 online courses to supplement learning',
            'action': 'Explore platforms like Coursera, edX, or Khan Academy'
        })
    
    # Stress Level suggestions
    if stress_level == '2':  # High stress
        suggestions.append({
            'priority': 'high',
            'category': 'Stress Management',
            'current': 'High',
            'recommendation': 'High stress levels can impact performance - consider stress management techniques',
            'action': 'Practice meditation, exercise regularly, and maintain work-life balance'
        })
    elif stress_level == '0':  # Low stress
        suggestions.append({
            'priority': 'low',
            'category': 'Stress Management',
            'current': 'Low',
            'recommendation': 'Good stress levels! Keep maintaining balance',
            'action': 'Continue your current stress management practices'
        })
    
    # Resources suggestions
    if resources == '0':  # No resources
        suggestions.append({
            'priority': 'high',
            'category': 'Learning Resources',
            'current': 'No Resources',
            'recommendation': 'Access to learning resources is crucial for success',
            'action': 'Utilize library resources, online materials, and study groups'
        })
    
    # Internet access suggestions
    if internet == '0':  # No internet
        suggestions.append({
            'priority': 'high',
            'category': 'Internet Access',
            'current': 'No Internet',
            'recommendation': 'Internet access is essential for online learning and research',
            'action': 'Consider using campus facilities or public WiFi spots'
        })
    
    # Result-specific suggestions
    if prediction_result == 'Fail':
        suggestions.append({
            'priority': 'high',
            'category': 'Overall Performance',
            'current': 'Fail',
            'recommendation': 'Comprehensive improvement needed across all areas',
            'action': 'Meet with academic advisors and create a detailed improvement plan'
        })
    elif prediction_result == 'Satisfactory':
        suggestions.append({
            'priority': 'medium',
            'category': 'Overall Performance',
            'current': 'Satisfactory',
            'recommendation': 'Good foundation, but room for improvement',
            'action': 'Focus on weak areas and aim for consistent performance'
        })
    elif prediction_result == 'Good':
        suggestions.append({
            'priority': 'low',
            'category': 'Overall Performance',
            'current': 'Good',
            'recommendation': 'Solid performance! Keep up the good work',
            'action': 'Maintain current habits and aim for excellence'
        })
    
    # Sort suggestions by priority
    priority_order = {'high': 1, 'medium': 2, 'low': 3}
    suggestions.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    return suggestions

model = joblib.load("best_random_forest_model.pkl")
feature_names = ['StudyHours', 'Attendance', 'Resources', 'Internet', 'Gender', 'Age', 'AssignmentCompletion', 'OnlineCourses', 'StressLevel']

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = pd.DataFrame([[
            float(request.form["StudyHours"]),
            float(request.form["Attendance"]),
            float(request.form["Resources"]),
            float(request.form["Internet"]),
            float(request.form["Gender"]),
            float(request.form["Age"]),
            float(request.form["AssignmentCompletion"]),
            float(request.form["OnlineCourses"]),
            float(request.form["StressLevel"])
        ]], columns=feature_names)
        

        prediction = model.predict(input_data)

        if prediction[0] == 0:
            result = "Excellent"
        elif prediction[0] == 1:
            result = "Good"
        elif prediction[0] == 2:
            result = "Satisfactory"  
        else:            
            result = "Fail"
        
        # Store result and details in session for PDF download
        entered_details = {name: request.form.get(name, "") for name in feature_names}
        readable_details = convert_details_to_readable(entered_details)
        
        # Generate improvement suggestions
        suggestions = generate_improvement_suggestions(entered_details, result)
        
        session['prediction_result'] = result
        session['prediction_details'] = readable_details
        session['improvement_suggestions'] = suggestions

        return render_template("results.html", prediction_text=result, details=readable_details, suggestions=suggestions)

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/download_result")
def download_result():
    """Generate and download prediction result as PDF"""
    result = session.get('prediction_result', 'Unknown')
    details = session.get('prediction_details', {})
    
    if not result or not details:
        return "No prediction data available. Please run a prediction first.", 400
    
    # Create PDF in memory
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Build document elements
    elements = []
    
    # Title
    elements.append(Paragraph("Exam Prediction Result", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Result Section
    elements.append(Paragraph("Prediction Result", heading_style))
    result_color = {
        "Excellent": colors.HexColor('#27ae60'),
        "Good": colors.HexColor('#3498db'),
        "Satisfactory": colors.HexColor('#f39c12'),
        "Fail": colors.HexColor('#e74c3c')
    }.get(result, colors.black)
    
    result_style = ParagraphStyle(
        'ResultText',
        parent=styles['Normal'],
        fontSize=18,
        textColor=result_color,
        fontName='Helvetica-Bold'
    )
    elements.append(Paragraph(result, result_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Details Section
    elements.append(Paragraph("Entered Details", heading_style))
    
    # Create table for details
    detail_data = [["Parameter", "Value"]]
    for key, value in details.items():
        detail_data.append([key, str(value)])
    
    detail_table = Table(detail_data, colWidths=[3*inch, 2*inch])
    detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer with date
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=1
    )
    elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'exam_prediction_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )

if __name__ == "__main__":
    app.run(debug=True)