from flask import Flask, request, jsonify
from recommendation import get_recommendations

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the AI-powered Recommendation Engine API!"

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Please provide user_id as query param'}), 400
    
    try:
        user_id = int(user_id)
    except:
        return jsonify({'error': 'user_id must be an integer'}), 400

    recommendations = get_recommendations(user_id)
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
