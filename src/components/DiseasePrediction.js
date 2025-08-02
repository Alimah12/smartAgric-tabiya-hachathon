import React, { useState } from 'react';
import axios from 'axios';
import './DiseasePrediction.css'; // Optional CSS for styling

function DiseasePrediction() {
    const [file, setFile] = useState(null);
    const [predictedDisease, setPredictedDisease] = useState('');
    const [advice, setAdvice] = useState(''); // New state for advice
    const [imagePreview, setImagePreview] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
        if (selectedFile) {
            setImagePreview(URL.createObjectURL(selectedFile));
            setError(''); // Clear any previous error
        } else {
            setImagePreview('');
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(''); // Clear previous error
        setPredictedDisease(''); // Clear previous prediction
        setAdvice(''); // Clear previous advice
        setLoading(true); // Set loading state

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5001/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            // Check for predicted disease and advice
            if (response.data.predicted_disease) {
                setPredictedDisease(response.data.predicted_disease);
                setAdvice(response.data.advice || 'No advice available'); // Set the advice
            } else {
                setError('No disease predicted');
            }
        } catch (error) {
            console.error('Error predicting disease:', error);
            setError('Error predicting disease: ' + (error.response?.data?.error || 'Please try again.'));
        } finally {
            setLoading(false); // Reset loading state
        }
    };

    return (
        <div className="disease-prediction" style={{ height: '100vh', overflow: 'auto', padding: '20px', width: '90%', maxWidth: '1200px', margin: '0 auto', marginTop: '60px' }}> {/* Increased margin for space */}
            <h2 style={{ textAlign: 'center', marginTop: '20px' }}>Disease Prediction</h2>

            {/* Description Section */}
            <div className="description-section" style={{ margin: '40px 0', textAlign: 'center' }}>
                <p style={{ fontSize: '18px', marginBottom: '20px' }}>
                    Discover potential diseases affecting your crops with our advanced image recognition technology. 
                    Simply upload a leaf image, and our system will analyze it to provide accurate predictions 
                    and actionable advice.
                </p>
                {/* Using the online image link */}
                <img 
                    src="https://media.istockphoto.com/id/1656285526/de/foto/tomaten-krankheiten.jpg?s=612x612&w=0&k=20&c=uKpajBvIca0Jgt0C5wzOcYpZELYIURTlOfBtIxhhT6k=" 
                    alt="Crops" 
                    style={{ width: '80%', height: 'auto', borderRadius: '10px' }} 
                />
            </div>

            <form onSubmit={handleSubmit} style={{ margin: '60px auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}> {/* Centering the form elements */}
                <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileChange} 
                    required 
                    style={{ marginBottom: '20px', width: '100%', maxWidth: '400px' }} // Set width for the input
                />
                {imagePreview && <img src={imagePreview} alt="Leaf Preview" className="image-preview" style={{ width: '100%', marginBottom: '20px' }} />}
                <button type="submit" disabled={!file || loading} style={{ padding: '10px 20px', fontSize: '16px' }}>
                    {loading ? 'Predicting...' : 'Predict Disease'}
                </button>
            </form>

            <div style={{ marginTop: '60px', textAlign: 'center' }}> {/* Center the results */}
                {predictedDisease && <h3>Predicted Disease: {predictedDisease}</h3>}
                {advice && <p><strong>Advice:</strong> {advice}</p>} {/* Display the advice */}
                {error && <h3 className="error-message">{error}</h3>}
            </div>
        </div>
    );
}

export default DiseasePrediction;
