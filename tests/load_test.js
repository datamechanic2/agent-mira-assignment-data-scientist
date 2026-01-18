import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = 'https://agent-mira-assignment-data-scientist.onrender.com';

export const options = {
  stages: [
    { duration: '10s', target: 5 },   // ramp up to 5 users
    { duration: '20s', target: 10 },  // ramp up to 10 users
    { duration: '10s', target: 0 },   // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],     // less than 10% failures
  },
};

const testProperties = [
  { location: 'CityA', size: 2500, bedrooms: 3, bathrooms: 2, year_built: 2010, condition: 'Good', property_type: 'Single Family' },
  { location: 'CityB', size: 1800, bedrooms: 2, bathrooms: 1, year_built: 2015, condition: 'New', property_type: 'Condominium' },
  { location: 'CityC', size: 3200, bedrooms: 4, bathrooms: 3, year_built: 2005, condition: 'Fair', property_type: 'Townhouse' },
  { location: 'CityA', size: 4000, bedrooms: 5, bathrooms: 4, year_built: 2020, condition: 'New', property_type: 'Single Family' },
];

export default function () {
  // health check
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, {
    'health status 200': (r) => r.status === 200,
    'model loaded': (r) => JSON.parse(r.body).model_loaded === true,
  });

  // prediction request
  const property = testProperties[Math.floor(Math.random() * testProperties.length)];
  const predictRes = http.post(
    `${BASE_URL}/predict`,
    JSON.stringify(property),
    { headers: { 'Content-Type': 'application/json' } }
  );

  check(predictRes, {
    'predict status 200': (r) => r.status === 200,
    'has predicted_price': (r) => JSON.parse(r.body).predicted_price !== undefined,
    'has model_version': (r) => JSON.parse(r.body).model_version !== undefined,
  });

  sleep(0.5);
}
