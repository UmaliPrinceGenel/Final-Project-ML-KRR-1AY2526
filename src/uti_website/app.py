import os
import re
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ==========================================
# 1. LOAD THE MODEL
# ==========================================
base_dir = os.path.abspath(os.path.dirname(__file__))

def get_path(filename):
    return os.path.join(base_dir, filename)

model = None
model_columns = None
color_mode_val = None
custom_threshold = 0.8518

try:
    model = joblib.load(get_path('uti_model.pkl'))
    model_columns = joblib.load(get_path('model_columns.pkl'))
    color_mode_val = joblib.load(get_path('color_mode.pkl'))
    custom_threshold = joblib.load(get_path('uti_threshold.pkl'))
    print(f"Loaded Model from: {base_dir}")
except FileNotFoundError as e:
    print("\nCRITICAL PATH ERROR")
    print(f"Python is looking in: {base_dir}")
    print(f"But it could not find: {e.filename}")

# ==========================================
# 2. UNIFIED DATA PARSING (IMPROVED)
# ==========================================
class UrinalysisParser:
    @staticmethod
    def parse_numeric(val):
        """
        Unified parser for all numeric/range inputs.
        Handles: '10-12', 'TNTC', 'Trace', '1+', 'Negative', 'Loaded'
        """
        if pd.isna(val) or val == '': 
            return 0.0
            
        if isinstance(val, (int, float)):
            return float(val)

        val_str = str(val).upper().strip()
        
        # 1. Handle Text Mappings
        text_map = {
            'NEGATIVE': 0.0, 'NONE': 0.0, 'NONE SEEN': 0.0,
            'TRACE': 0.5, 'RARE': 1.0, 
            'OCCASIONAL': 2.0, 'FEW': 3.0, 
            'MODERATE': 4.0, 'MANY': 6.0,
            'PLENTY': 7.0, 'LOADED': 8.0, 
            'TNTC': 100.0,  
            '1+': 1.0, '2+': 2.0, '3+': 3.0, '4+': 4.0
        }
        
        if val_str in text_map:
            return text_map[val_str]
            
        # 2. Handle Ranges
        if '-' in val_str:
            try:
                parts = val_str.split('-')
                # Take average of the first two valid numbers found
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                pass 

        # 3. Handle Direct Numbers / Fallback
        try:
            return float(val_str)
        except ValueError:
            digits = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
            return float(digits[0]) if digits else 0.0

def clean_input_data(input_dict):

    df = pd.DataFrame([input_dict])
    
    cols_to_convert = ['Age', 'pH', 'Specific Gravity', 'Glucose', 'Protein']
    for col in cols_to_convert:
        # Use parser to handle 'Trace', '1+', etc. 
        val = df.get(col, pd.Series([0])).iloc[0]
        df[col] = UrinalysisParser.parse_numeric(val)

    # --- B. Parse Microscope Ranges ---
    microscope_cols = ['WBC', 'RBC', 'Epithelial Cells', 'Mucous Threads', 'Amorphous Urates', 'Bacteria']
    for col in microscope_cols:
        val = df.get(col, pd.Series([0])).iloc[0]
        df[col] = UrinalysisParser.parse_numeric(val)

    # --- C. Handle Categorical Defaults ---
    if color_mode_val is not None:
        if 'Color' not in df.columns or pd.isna(df['Color'][0]) or df['Color'][0] == '':
            df['Color'] = color_mode_val
    else:
        df['Color'] = 'YELLOW'

    # --- D. One-Hot Encoding for ML Model ---
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Color', 'Transparency'])

    if model_columns is not None:
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    else:
        df_final = df_encoded

    # --- E. Prepare Clean Dictionary for Expert System ---
    raw_cleaned_data = input_dict.copy()
    for col in cols_to_convert + microscope_cols:
        raw_cleaned_data[col] = float(df[col].iloc[0])
    
    return df_final, raw_cleaned_data

# ==========================================
# 3. RULE-BASED LOGIC 
# ==========================================
def calculate_dynamic_probabilities(row_dict):
    gender = str(row_dict.get('Gender', 'FEMALE')).upper()
    
    conditions = {
        'Kidney_Disease':      {'score': 0, 'ceiling': 80},
        'Diabetes':            {'score': 0, 'ceiling': 50},
        'Liver_Issues':        {'score': 0, 'ceiling': 40},
        'Dehydration':         {'score': 0, 'ceiling': 50},
        'Hematuria':           {'score': 0, 'ceiling': 50},
        'Sample_Contamination':{'score': 0, 'ceiling': 40} 
    }
    
    reasons = {k: [] for k in conditions.keys()}
    reasons['Normal'] = []

    # Parse ALL features
    age = row_dict.get('Age', 0)
    prot = row_dict.get('Protein', 0)
    gluc = row_dict.get('Glucose', 0)
    rbc = row_dict.get('RBC', 0)
    wbc = row_dict.get('WBC', 0)        
    ph = row_dict.get('pH', 0)
    sg = row_dict.get('Specific Gravity', 0)
    epith = row_dict.get('Epithelial Cells', 0)
    mucous = row_dict.get('Mucous Threads', 0)
    urates = row_dict.get('Amorphous Urates', 0)
    bacteria = row_dict.get('Bacteria', 0)
    color = str(row_dict.get('Color', '')).upper()
    transparency = str(row_dict.get('Transparency', '')).upper()

    
    # Kidney Disease
    if wbc >= 2:
        score = min(wbc * 3, 30)
        conditions['Kidney_Disease']['score'] += score
        if wbc >= 10: reasons['Kidney_Disease'].append(f"Very high WBC ({wbc}) suggests severe infection.")
        else: reasons['Kidney_Disease'].append(f"Elevated WBC ({wbc}) indicates inflammation.")
        if gender == 'MALE' and wbc >= 4:
            conditions['Kidney_Disease']['score'] += 10
            reasons['Kidney_Disease'].append("Significant pyuria in male warrants closer evaluation.")

    if prot >= 1:
        score = prot * 12
        conditions['Kidney_Disease']['score'] += score
        reasons['Kidney_Disease'].append(f"Proteinuria ({prot}+) suggests glomerular dysfunction.")
        if prot >= 3: reasons['Kidney_Disease'].append("Severe proteinuria (≥3+) indicates significant renal damage.")

    if bacteria >= 2:
        conditions['Kidney_Disease']['score'] += 12
        reasons['Kidney_Disease'].append(f"Bacteriuria present (level {bacteria}).")
        if wbc >= 2 and bacteria >= 3:
            conditions['Kidney_Disease']['score'] += 12
            reasons['Kidney_Disease'].append("Combined bacteriuria + pyuria confirms active infection.")

    if age >= 50:
        age_score = min((age - 50) * 0.3, 8)
        conditions['Kidney_Disease']['score'] += age_score
        if age >= 65: reasons['Kidney_Disease'].append(f"Advanced age ({int(age)}) increases kidney disease risk.")

    # Diabetes
    if gluc >= 1:
        score = gluc * 20
        conditions['Diabetes']['score'] += score
        reasons['Diabetes'].append(f"Glycosuria detected ({gluc}+).")
        if ph > 0 and ph < 6.0:
            conditions['Diabetes']['score'] += 10
            reasons['Diabetes'].append(f"Low pH ({ph:.1f}) with glucose may indicate ketoacidosis.")

    # Hematuria
    if rbc > 2:
        score = min(rbc * 3, 50)
        conditions['Hematuria']['score'] += score
        reasons['Hematuria'].append(f"Microscopic hematuria detected (RBC: {rbc}).")
        if rbc > 15:
            conditions['Kidney_Disease']['score'] += 15
            reasons['Kidney_Disease'].append("Severe hematuria suggests renal injury, stones, or malignancy.")

    if color in ['RED', 'LIGHT RED', 'REDDISH', 'REDDISH YELLOW', 'DARK RED']:
        conditions['Hematuria']['score'] += 25
        reasons['Hematuria'].append(f"Visible hematuria (color: {color}).")
        if rbc <= 2:
            conditions['Kidney_Disease']['score'] += 8
            reasons['Kidney_Disease'].append("Visible blood with low RBC count suggests intermittent bleeding.")

    # Liver Issues
    if color in ['BROWN', 'AMBER', 'DARK YELLOW']:
        conditions['Liver_Issues']['score'] += 30
        reasons['Liver_Issues'].append(f"Abnormal color ({color}) suggests bilirubin presence.")

    # Dehydration
    if sg >= 1.025:
        if gluc >= 2:
             reasons['Dehydration'].append(f"Note: High Specific Gravity ({sg:.3f}) may be caused by Glucose.")
             score = 10 
        else:
            if sg >= 1.035:
                score = 40; reasons['Dehydration'].append(f"Very high specific gravity ({sg:.3f}) indicates severe dehydration.")
            elif sg >= 1.030:
                score = 28; reasons['Dehydration'].append(f"High specific gravity ({sg:.3f}) indicates significant dehydration.")
            else:
                score = 18; reasons['Dehydration'].append(f"Elevated specific gravity ({sg:.3f}) suggests mild dehydration.")
        conditions['Dehydration']['score'] += score
    
    if age >= 60 and conditions['Dehydration']['score'] > 0:
        conditions['Dehydration']['score'] += 8
        reasons['Dehydration'].append(f"Older age ({int(age)}) increases dehydration risk.")

    if urates >= 3:
        conditions['Dehydration']['score'] += 10
        reasons['Dehydration'].append("Amorphous urates support concentrated urine finding.")

    # Kidney Stones
    if ph >= 8.0:
        conditions['Kidney_Disease']['score'] += 12
        reasons['Kidney_Disease'].append(f"Alkaline pH ({ph:.1f}) increases stone formation risk.")
    
    if transparency in ['TURBID', 'CLOUDY']:
        if bacteria < 2 and wbc < 2:
            conditions['Kidney_Disease']['score'] += 8
            reasons['Kidney_Disease'].append("Turbidity without infection suggests crystals/stones.")

    # Sample Contamination
    contamination_detected = False
    if epith >= 3 or mucous >= 3:
        contamination_score = (epith * 5) + (mucous * 3)
        if gender == 'FEMALE':
            contamination_score *= 0.75
            reasons['Sample_Contamination'].append("Elevated epithelial cells (common in female samples, but high).")
        else:
            reasons['Sample_Contamination'].append("High epithelial cells in male sample indicates contamination.")
        conditions['Sample_Contamination']['score'] += contamination_score
        contamination_detected = True
        if prot >= 1 and epith >= 5:
            reasons['Kidney_Disease'].append("Note: Protein may be inflated by contamination. Recommend repeat.")

    # --- 2. CALCULATE FINAL PROBABILITIES ---
    final_probabilities = {}
    for cond, data in conditions.items():
        if data['ceiling'] > 0:
            raw_prob = (data['score'] / data['ceiling']) * 100
            prob = min(round(raw_prob, 1), 99.9)
            if prob > 15:
                final_probabilities[cond] = prob

    normal_indicators = 0
    normal_total = 7  
    
    if wbc < 2: normal_indicators += 1
    if rbc < 3: normal_indicators += 1
    if prot < 1: normal_indicators += 1
    if bacteria < 2: normal_indicators += 1
    if 5.0 <= ph <= 7.5: normal_indicators += 1
    

    if gluc < 1: normal_indicators += 1       
    if 1.005 <= sg <= 1.030: normal_indicators += 1  
    
    normal_prob = round((normal_indicators / normal_total) * 100, 1)
    
    
    medical_risks = {k:v for k,v in final_probabilities.items() 
                     if k not in ['Sample_Contamination', 'Normal']}
    
    highest_risk_val = 0
    highest_risk_name = None
    
    if medical_risks:
        highest_risk_name = max(medical_risks, key=medical_risks.get)
        highest_risk_val = medical_risks[highest_risk_name]

    if highest_risk_val > 50:
        primary_condition = highest_risk_name
        
     
        if 'Normal' in final_probabilities:
            del final_probabilities['Normal']

    elif normal_prob >= 60:
        final_probabilities['Normal'] = normal_prob
        primary_condition = 'Normal'
        reasons['Normal'].append(f"Most urinalysis parameters within normal range ({normal_indicators}/{normal_total}).")
        
    else:
        primary_condition = 'Sample_Contamination' if contamination_detected else 'Indeterminate'

   # --- 5. GENERATE ADVISORY ---
    advisory = "Findings: "
    if primary_condition == 'Normal':
        advisory += "Urinalysis parameters within normal limits. "
    elif primary_condition == 'Indeterminate':
        advisory += "Several Borderline findings detected. No specific disease pattern is strong enough to confirm, but results are not fully normal. Clinical correlation recommended. "
    else:
        high_risks = [k for k, v in final_probabilities.items() 
                     if v > 50 and k not in ['Normal', 'Sample_Contamination']]
        if high_risks:
            advisory += f"Strong indicators for: {', '.join(high_risks)}. "
        else:
            advisory += f"Possible {primary_condition}. "
    
    if final_probabilities.get('Sample_Contamination', 0) > 40:
        advisory += "⚠ WARNING: Sample quality concern. Recommend repeat collection with proper technique. "

    return {
        'probabilities': final_probabilities, 
        'reasons': reasons, 
        'primary_condition': primary_condition, 
        'advisory': advisory
    }

# ==========================================
# 4. IMPROVED UTI ADVISORY
# ==========================================
def generate_uti_advisory(prediction, probability, row_data):
    wbc_val = row_data.get('WBC', 0)
    rbc_val = row_data.get('RBC', 0)
    protein_val = row_data.get('Protein', 0)
    bacteria_val = row_data.get('Bacteria', 0)
    color = str(row_data.get('Color', '')).upper()
    transparency = str(row_data.get('Transparency', '')).upper()
    
    uti_indicators = 0
    indicator_details = []
    
    if wbc_val >= 5:
        uti_indicators += 2
        indicator_details.append(f"Elevated WBC ({wbc_val})")
    elif wbc_val >= 2:
        uti_indicators += 1
        indicator_details.append(f"Mild WBC elevation ({wbc_val})")
    
    if bacteria_val >= 3:
        uti_indicators += 2
        indicator_details.append(f"Significant bacteriuria ({bacteria_val})")
    elif bacteria_val >= 2:
        uti_indicators += 1
        indicator_details.append(f"Bacteria present ({bacteria_val})")
    
    if rbc_val >= 5:
        uti_indicators += 1
        indicator_details.append(f"Hematuria ({rbc_val} RBC)")
    
    if protein_val >= 1:
        uti_indicators += 1
        indicator_details.append(f"Proteinuria ({protein_val}+)")
    
    if transparency in ['TURBID', 'CLOUDY']:
        uti_indicators += 1
        indicator_details.append("Turbid appearance")

    hematuria_colors = ['RED', 'LIGHT RED', 'REDDISH', 'REDDISH YELLOW']    
    if color in hematuria_colors:
        uti_indicators += 1
        indicator_details.append(f"Abnormal Color ({color}) suggests hematuria")

    conf_pct = probability * 100
    advisory = {
        "status": "", 
        "confidence": f"{conf_pct:.1f}%", 
        "severity": "", 
        "recommendation": "",
        "supporting_evidence": ", ".join(indicator_details) if indicator_details else "Limited"
    }

    if prediction == 1:
        advisory['status'] = "POSITIVE FOR UTI"
        
        if conf_pct > 90:
            advisory['severity'] = "High Confidence"
            action = "Immediate medical consultation recommended."
        elif conf_pct > 70:
            advisory['severity'] = "Moderate Confidence"
            action = "Medical consultation advised within 24-48 hours."
        else:
            advisory['severity'] = "Low Confidence (Borderline)"
            action = "Clinical correlation needed. Monitor symptoms."

        # Adjust recommendation based on supporting evidence
        if uti_indicators >= 4:
            advisory['recommendation'] = f"{action} Strong laboratory evidence (multiple abnormal findings). Urine culture recommended."
        elif uti_indicators >= 2:
            advisory['recommendation'] = f"{action} Moderate laboratory support. Consider urine culture if symptomatic."
        else:
            advisory['recommendation'] = f"{action} Limited supporting evidence - result may be false positive. Clinical symptoms are critical for diagnosis."

    else:
        advisory['status'] = "NEGATIVE FOR UTI"
        
        if conf_pct < 10:
            advisory['severity'] = "High Confidence - Normal"
            advisory['recommendation'] = "No evidence of UTI. All parameters within normal limits."
        elif conf_pct < 30:
            advisory['severity'] = "Low UTI Risk"
            advisory['recommendation'] = "UTI unlikely based on urinalysis. Maintain hydration."
        elif conf_pct < 50:
            advisory['severity'] = "Borderline"
            advisory['recommendation'] = "Some abnormal findings present but below UTI threshold. Monitor for symptom development."
        else:
            advisory['severity'] = "Indeterminate"
            advisory['recommendation'] = "Conflicting results. If symptomatic, consider repeat urinalysis or urine culture."

    return advisory


# ==========================================
# 5. WEB ROUTES
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            
            # A. Clean Data (Returns ML inputs AND Cleaned Row for logic)
            X_input, raw_clean_row = clean_input_data(form_data)

            if model:
                # B. ML Prediction (Using CUSTOM THRESHOLD)
                prob = model.predict_proba(X_input)[0][1]
                pred = 1 if prob >= custom_threshold else 0
            else:
                # Fallback if model failed to load
                prob = 0.0
                pred = 0

            # C. Expert System (Uses the Unified 'raw_clean_row')
            rule_results = calculate_dynamic_probabilities(raw_clean_row)
            uti_advice = generate_uti_advisory(pred, prob, raw_clean_row)
            
            return render_template('index.html', 
                                   prediction=pred, 
                                   probability=round(prob*100, 1),
                                   uti_advice=uti_advice,
                                   rule_results=rule_results)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"An error occurred while processing your data: {e}"
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)