// src/components/CropYield.js
import React, { useState } from 'react';
import './CropYieldPrediction.css'; // Add custom styles for CropYield

const CropYield = () => {
    const [location, setLocation] = useState('');
    const [landSize, setLandSize] = useState('');
    const [climateData, setClimateData] = useState(null);
    const [recommendedCrops, setRecommendedCrops] = useState(null);
    const [yieldData, setYieldData] = useState(null);

    // Hardcoded location-based climate data and crop recommendations
    const locationData = {
        "Mombasa": {
            rainfall: '600 mm',
            soilPH: '6.5',
            temperature: '28°C',
            crops: [
                { name: 'Maize', yield: 3, tips: 'Plant during the long rains for optimal growth.' },
                { name: 'Cassava', yield: 2.5, tips: 'Grows well in sandy soil; minimal watering required.' },
                { name: 'Mangoes', yield: 1.8, tips: 'Plant grafted varieties for high yield.' }
            ]
        },
        "Nakuru": {
            rainfall: '800 mm',
            soilPH: '6.8',
            temperature: '20°C',
            crops: [
                { name: 'Wheat', yield: 4, tips: 'Ensure timely weeding and top-dress with nitrogen.' },
                { name: 'Barley', yield: 3.5, tips: 'Requires consistent rainfall; avoid waterlogging.' },
                { name: 'Potatoes', yield: 6, tips: 'Requires well-drained soil and regular earthing-up.' }
            ]
        },
        "Garissa": {
            rainfall: '300 mm',
            soilPH: '7.2',
            temperature: '32°C',
            crops: [
                { name: 'Sorghum', yield: 1.5, tips: 'Drought-resistant crop; thrives in low rainfall areas.' },
                { name: 'Millet', yield: 1.2, tips: 'Ensure good seed selection for better yields.' },
                { name: 'Dates', yield: 2, tips: 'Requires hot climate and minimal water for growth.' }
            ]
        }
    };

    // Handle the process of setting climate data and recommending crops
    const handleLocationChange = (e) => {
        const selectedLocation = e.target.value;
        setLocation(selectedLocation);

        if (locationData[selectedLocation]) {
            setClimateData(locationData[selectedLocation]);
            setRecommendedCrops(locationData[selectedLocation].crops);
        } else {
            setClimateData(null);
            setRecommendedCrops(null);
        }
    };

    // Calculate yield based on the selected crop and land size
    const handleCalculateYield = () => {
        if (landSize && recommendedCrops) {
            const yieldResults = recommendedCrops.map(crop => ({
                ...crop,
                totalYield: crop.yield * parseFloat(landSize)
            }));
            setYieldData(yieldResults);
        }
    };

    return (
        <div className="crop-yield-container" style={{ height: '100vh', overflow: 'auto', padding: '20px', width: '90%', maxWidth: '1200px', margin: '0 auto', marginTop: '60px' }}>
            <h1 style={{ marginTop: '40px', textAlign: 'center' }}>Crop Yield Prediction</h1>
            <p style={{ textAlign: 'center' }}>Input your location and land size to get crop yield predictions and management tips.</p>

            {/* Location and land size form */}
            <div className="yield-form" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <label htmlFor="locationSelect">Select Location:</label>
                <select id="locationSelect" value={location} onChange={handleLocationChange} style={{ marginBottom: '20px', padding: '10px', width: '100%', maxWidth: '400px' }}>
                    <option value="">-- Select Location --</option>
                    <option value="Mombasa">Mombasa</option>
                    <option value="Nakuru">Nakuru</option>
                    <option value="Garissa">Garissa</option>
                </select>

                {climateData && (
                    <div className="climate-info" style={{ textAlign: 'center', marginBottom: '20px' }}>
                        <h3>Climate Data for {location}</h3>
                        <ul style={{ listStyle: 'none', padding: 0 }}>
                            <li><strong>Rainfall:</strong> {climateData.rainfall}</li>
                            <li><strong>Soil pH:</strong> {climateData.soilPH}</li>
                            <li><strong>Temperature:</strong> {climateData.temperature}</li>
                        </ul>
                    </div>
                )}

                <label htmlFor="landSize">Enter Land Size (in acres):</label>
                <input
                    type="number"
                    id="landSize"
                    value={landSize}
                    onChange={(e) => setLandSize(e.target.value)}
                    placeholder="e.g., 2"
                    style={{ marginBottom: '20px', padding: '10px', width: '100%', maxWidth: '400px' }}
                />

                <button onClick={handleCalculateYield} style={{ padding: '10px 20px', fontSize: '16px' }}>Calculate Yield</button>
            </div>

            {/* Display recommended crops */}
            {recommendedCrops && (
                <div className="recommended-crops" style={{ textAlign: 'center', marginTop: '40px' }}>
                    <h2>Recommended Crops for {location}</h2>
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                        {recommendedCrops.map((crop, index) => (
                            <li key={index}>
                                <strong>{crop.name}:</strong> {crop.tips}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Display yield results */}
            {yieldData && (
                <div className="yield-result" style={{ textAlign: 'center', marginTop: '40px' }}>
                    <h2>Yield Predictions</h2>
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                        {yieldData.map((crop, index) => (
                            <li key={index}>
                                <strong>{crop.name}:</strong> {crop.totalYield} tonnes for {landSize} acres.
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default CropYield;
