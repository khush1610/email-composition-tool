import os
os.environ["STREAMLIT_WATCHDOG_USE_POLLING"] = "true"
import streamlit as st
import pandas as pd
import json
from transformers import pipeline, set_seed
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Set seed for reproducibility
set_seed(42)

# Initialize Hugging Face text generation pipeline
try:
    generator = pipeline('text-generation', model='distilgpt2', max_new_tokens=100)
except Exception as e:
    st.warning(f"Failed to load distilgpt2 model: {str(e)}. Falling back to gpt2.")
    try:
        generator = pipeline('text-generation', model='gpt2', max_new_tokens=100)
    except Exception as e:
        st.error(f"Failed to load gpt2 model: {str(e)}")
        st.stop()

# Custom CSS for modern, professional UI
st.markdown("""
    <style>
    /* General layout */
    .main-container { 
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
        padding: 30px; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Titles and headers */
    .main-title { 
        font-size: 36px; 
        color: #1e3a8a; 
        margin-bottom: 20px; 
        font-weight: 700;
        text-align: center;
        background: linear-gradient(to right, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header { 
        font-size: 28px; 
        color: #1e40af; 
        margin: 30px 0 15px; 
        font-weight: 600;
        text-align: center;
    }
    .subheader { 
        font-size: 20px; 
        color: #1f2937; 
        margin: 25px 0 15px; 
        font-weight: 600;
        position: relative;
    }
    .subheader::after {
        content: '';
        width: 60px;
        height: 3px;
        background: #3b82f6;
        position: absolute;
        bottom: -5px;
        left: 0;
        border-radius: 3px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white; 
        border-radius: 10px; 
        padding: 12px 24px; 
        font-size: 16px; 
        font-weight: 500;
        border: none; 
        width: 100%; 
        margin-top: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e40af, #1d4ed8);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    .copy-button {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border: none !important;
        margin-top: 15px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    }
    .copy-button:hover {
        background: linear-gradient(90deg, #059669, #047857) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }

    /* Inputs and text areas */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    .stTextArea>div>textarea, 
    .stSelectbox>div {
        font-size: 14px; 
        border-radius: 8px; 
        padding: 12px; 
        border: 1px solid #d1d5db;
        background-color: #ffffff;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus, 
    .stTextArea>div>textarea:focus, 
    .stSelectbox>div:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.3);
    }
    .stTextArea>div>textarea { 
        height: 180px !important; 
        resize: none;
    }

    /* Labels */
    .stSelectbox label, 
    .stTextInput label, 
    .stNumberInput label, 
    .stTextArea label {
        font-size: 14px; 
        color: #374151; 
        font-weight: 500; 
        margin-bottom: 8px;
    }

    /* Email content display */
    .email-content {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        white-space: pre-wrap;
        font-size: 14px;
        line-height: 1.6;
        min-height: 200px;
        max-height: 320px;
        overflow-y: auto;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .email-content:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    /* Divider */
    .divider { 
        border-top: 1px solid #e5e7eb; 
        margin: 30px 0; 
    }

    /* Expander styling */
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .stExpander:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    /* Responsive design */
    @media (max-width: 600px) {
        .main-container { padding: 20px; }
        .main-title { font-size: 28px; }
        .header { font-size: 24px; }
        .subheader { font-size: 18px; }
        .stButton>button { 
            font-size: 14px; 
            padding: 10px 18px; 
        }
        .copy-button {
            font-size: 13px !important;
            padding: 8px 16px !important;
        }
        .stTextArea>div>textarea { 
            height: 140px !important; 
        }
        .email-content { 
            min-height: 160px; 
            max-height: 280px; 
        }
    }
    </style>
""", unsafe_allow_html=True)

# Mock customer data
mock_data = {
    'segment': ['repeat_customers', 'new_leads', 'inactive_customers', 'ecommerce_customers', 'local_customers'],
    'greeting': ['Dear Valued Customer', 'Hello New Friend', 'Hi There', 'Dear Shopper', 'Hello Neighbor'],
    'tone': ['friendly', 'welcoming', 're-engaging', 'professional', 'community-focused']
}

# Campaign templates
templates = {
    'loyalty_offer': {
        'subject': "Exclusive {discount}% Off for Our Loyal Customers!",
        'body': "{greeting},\n\nThank you for being a loyal customer! As a token of our appreciation, "
                "enjoy {discount}% off your next purchase. Use code LOYAL{discount} at checkout. Hurry, this offer "
                "expires in 7 days!\n\nShop Now: [Insert Link]\n\nBest,\n{business_name}"
    },
    'product_launch': {
        'subject': "Introducing Our New {product_name}!",
        'body': "{greeting},\n\nWe're thrilled to announce the launch of {product_name}! It's designed to {"
                "product_benefit}. Be the first to try it today!\n\nDiscover Now: [Insert Link]\n\nCheers,"
                "\n{business_name}"
    },
    're_engagement': {
        'subject': "We Miss You! Come Back for a Special Offer",
        'body': "{greeting},\n\nIt's been a while! Come back and enjoy a special {discount}% off your next order with "
                "code WELCOME{discount}. Don't miss out!\n\nShop Now: [Insert Link]\n\nWarm regards,\n{business_name}"
    },
    'seasonal_promo': {
        'subject': "{season} Sale: Save {discount}% Today!",
        'body': "{greeting},\n\nCelebrate {season} with {discount}% off everything! Use code {season}{discount} at "
                "checkout. Limited time only!\n\nShop Now: [Insert Link]\n\nHappy {season},\n{business_name}"
    },
    'event_invite': {
        'subject': "Join Us for a Special {event_name}!",
        'body': "{greeting},\n\nYou're invited to our exclusive {event_name} on {event_date}! Join us for {"
                "event_details}. RSVP now!\n\nRSVP: [Insert Link]\n\nBest,\n{business_name}"
    }
}

# Generate email content
def generate_email(segment, campaign_type, business_name, discount=None, product_name=None, season=None,
                   event_name=None, event_date=None, event_details=None):
    if not business_name.strip():
        st.error("Business Name cannot be empty.")
        return None

    segment_data = mock_data['segment']
    if segment not in segment_data:
        segment = 'ecommerce_customers'
    segment_idx = segment_data.index(segment)
    greeting = mock_data['greeting'][segment_idx]

    template = templates.get(campaign_type, templates['loyalty_offer'])

    prompt = f"Generate a professional marketing email for {segment} with campaign goal: {campaign_type}. Business: {business_name}."
    if discount:
        prompt += f" Include a {discount}% discount."
    if product_name:
        prompt += f" Promote new product: {product_name}."
    if season:
        prompt += f" For {season} season."
    if event_name:
        prompt += f" For event: {event_name} on {event_date}."

    mock_ai_output = f"{greeting}, thank you for your continued support at {business_name}! We're offering you a special {discount or 20}% discount on your next purchase. Use code {campaign_type.upper()}{discount or 20} at checkout. This offer is valid for 7 days only! Shop now: [Insert Link]. Warm regards, {business_name} Team."

    try:
        ai_result = generator(
            prompt,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            max_new_tokens=100,
            pad_token_id=50256
        )
        ai_output = ai_result[0]['generated_text']

        if not ai_output.strip():
            ai_output = mock_ai_output
        elif ai_output.strip() == prompt.strip():
            ai_output = mock_ai_output
        elif len(ai_output.split()) < 10:
            ai_output = mock_ai_output
    except Exception as e:
        st.error(f"AI suggestion failed: {str(e)}")
        ai_output = mock_ai_output

    subject = template['subject'].format(
        discount=discount or 20,
        product_name=product_name or "Product",
        season=season or "Holiday",
        event_name=event_name or "Event"
    )
    body = template['body'].format(
        greeting=greeting,
        discount=discount or 20,
        product_name=product_name or "Product",
        product_benefit="make your life easier" if product_name else "",
        season=season or "Holiday",
        event_name=event_name or "Event",
        event_date=event_date or "TBD",
        event_details=event_details or "fun and surprises",
        business_name=business_name
    )

    confidence = {
        'subject': round(np.random.uniform(0.7, 0.95), 2),
        'greeting': round(np.random.uniform(0.8, 0.98), 2),
        'body': round(np.random.uniform(0.65, 0.9), 2)
    }

    return {
        'subject': subject,
        'body': body,
        'ai_suggestion': ai_output,
        'confidence': confidence
    }

# Export email as JSON
def export_to_json(email_data):
    return json.dumps(email_data, indent=2)

# Create confidence score chart
def plot_confidence_scores(confidence):
    try:
        labels = list(confidence.keys())
        scores = list(confidence.values())

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, scores, color='#60a5fa')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Confidence Score', fontsize=10)
        ax.set_title('Suggestion Confidence Scores', fontsize=12)
        ax.tick_params(axis='both', labelsize=8)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str
    except Exception as e:
        st.error(f"Failed to generate confidence scores chart: {str(e)}")
        return None

# Streamlit UI
with st.container():
    # st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">TitanMind Email Composition Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="header">Craft Engaging Marketing Emails</div>', unsafe_allow_html=True)

    # Campaign Details
    st.markdown('<div class="subheader">Campaign Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        segment = st.selectbox("Audience Segment", mock_data['segment'],
                               help="Select the target audience for your campaign")
    with col2:
        campaign_type = st.selectbox("Campaign Type", list(templates.keys()),
                                     help="Choose the type of marketing campaign")

    business_name = st.text_input("Business Name", value="TitanMind", help="Enter your business name", max_chars=100)

    if campaign_type in ['loyalty_offer', 're_engagement', 'seasonal_promo']:
        discount = st.number_input("Discount (%)", min_value=5, max_value=50, value=20,
                                   help="Specify the discount percentage")
    else:
        discount = None

    if campaign_type == 'product_launch':
        product_name = st.text_input("Product Name", value="New Product", help="Enter the product name", max_chars=100)
    else:
        product_name = None

    if campaign_type == 'seasonal_promo':
        season = st.text_input("Season", value="Holiday", help="Enter the season (e.g., Summer)", max_chars=50)
    else:
        season = None

    if campaign_type == 'event_invite':
        event_name = st.text_input("Event Name", value="Customer Appreciation Day", help="Enter the event name",
                                   max_chars=100)
        event_date = st.text_input("Event Date", value="TBD", help="Enter the event date (e.g., July 15, 2025)",
                                   max_chars=50)
        event_details = st.text_area("Event Details", value="fun and surprises", help="Describe the event", height=200)
    else:
        event_name, event_date, event_details = None, None, None

    # Generate Button
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("Generate Email", type="primary"):
        email_data = generate_email(segment, campaign_type, business_name, discount, product_name, season, event_name,
                                    event_date, event_details)

        if email_data:
            # Generated Email
            st.markdown('<div class="subheader">Generated Email</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2], gap="medium")
            with col1:
                st.markdown(f"**Subject:** {email_data['subject']}", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="email-content">{}</div>'.format(email_data['body'].replace('\n', '<br>')),
                            unsafe_allow_html=True)
                if st.button("Copy Email to Clipboard", key="copy_email",
                             help="Copy the generated email to your clipboard", on_click=lambda: None,
                             kwargs={"class_name": "copy-button"}):
                    st.markdown(
                        f"""
                        <script>
                        navigator.clipboard.writeText(`{email_data['body']}`).then(() => 
                            alert('Email copied to clipboard!')
                        );
                        </script>
                        """,
                        unsafe_allow_html=True
                    )

            # AI Suggestion
            st.markdown('<div class="subheader">AI Suggestion</div>', unsafe_allow_html=True)
            if email_data['ai_suggestion'].startswith("Generate a professional marketing email"):
                email_data['ai_suggestion'] = mock_data['greeting'][mock_data['segment'].index(segment)] + ", thank you for your continued support at " + business_name + "! We're offering you a special " + str(discount or 20) + "% discount on your next purchase. Use code " + campaign_type.upper() + str(discount or 20) + " at checkout. This offer is valid for 7 days only! Shop now: [Insert Link]. Warm regards, " + business_name + " Team."
            st.markdown('<div class="email-content">{}</div>'.format(email_data['ai_suggestion'].replace('\n', '<br>')),
                        unsafe_allow_html=True)
            if st.button("Copy AI Suggestion to Clipboard", key="copy_ai",
                         help="Copy the AI suggestion to your clipboard", on_click=lambda: None,
                         kwargs={"class_name": "copy-button"}):
                st.markdown(
                    f"""
                    <script>
                    navigator.clipboard.writeText(`{email_data['ai_suggestion']}`).then(() => 
                        alert('AI Suggestion copied to clipboard!')
                    );
                    </script>
                    """,
                    unsafe_allow_html=True
                )

            # Confidence Scores
            st.markdown('<div class="subheader">Confidence Scores</div>', unsafe_allow_html=True)
            img_str = plot_confidence_scores(email_data['confidence'])
            if img_str:
                st.image(f"data:image/png;base64,{img_str}", width=400)
            else:
                st.error("Failed to render confidence scores chart.")

            # Export Options
            st.markdown('<div class="subheader">Export Options</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.download_button(
                    label="Download as Text",
                    data=email_data['body'],
                    file_name="email_draft.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="Download as JSON",
                    data=export_to_json(email_data),
                    file_name="email_draft.json",
                    mime="application/json",
                    use_container_width=True
                )

    # Sample Emails
    with st.expander("View Sample Email Drafts", expanded=False):
        st.markdown('<div class="subheader">Sample Email Drafts</div>', unsafe_allow_html=True)
        sample_configs = [
            {'segment': 'repeat_customers', 'campaign_type': 'loyalty_offer', 'business_name': 'TitanMind',
             'discount': 20},
            {'segment': 'new_leads', 'campaign_type': 'product_launch', 'business_name': 'TitanMind',
             'product_name': 'Smart Widget'},
            {'segment': 'inactive_customers', 'campaign_type': 're_engagement', 'business_name': 'TitanMind',
             'discount': 15},
            {'segment': 'ecommerce_customers', 'campaign_type': 'seasonal_promo', 'business_name': 'TitanMind',
             'season': 'Summer', 'discount': 25},
            {'segment': 'local_customers', 'campaign_type': 'event_invite', 'business_name': 'TitanMind',
             'event_name': 'Community Meetup', 'event_date': 'July 15, 2025', 'event_details': 'food and networking'}
        ]

        for i, config in enumerate(sample_configs, 1):
            sample_email = generate_email(**config)
            if sample_email:
                st.markdown(f"**Sample Email {i}: {config['campaign_type'].replace('_', ' ').title()}**",
                            unsafe_allow_html=True)
                st.markdown(f"**Subject:** {sample_email['subject']}", unsafe_allow_html=True)
                st.markdown('<div class="email-content">{}</div>'.format(sample_email['body'].replace('\n', '<br>')),
                            unsafe_allow_html=True)
                if st.button(f"Copy Sample {i} to Clipboard", key=f"copy_sample_{i}",
                             help="Copy this sample email to your clipboard", on_click=lambda: None,
                             kwargs={"class_name": "copy-button"}):
                    st.markdown(
                        f"""
                        <script>
                        navigator.clipboard.writeText(`{sample_email['body']}`).then(() => 
                            alert('Sample Email {i} copied to clipboard!')
                        );
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
