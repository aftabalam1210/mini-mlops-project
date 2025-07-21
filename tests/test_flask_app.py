import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up the test client once for all tests in this class
        cls.client = app.test_client()

    def test_home_page(self):
        """
        Tests that the home page loads correctly.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        """
        Tests the prediction functionality.
        It sends a POST request to /predict and follows the redirect
        to check the final result page.
        """
        # The key change is adding follow_redirects=True.
        # This tells the test client to handle the 302 redirect and load the final page.
        response = self.client.post('/predict', data={'text': "I love this!"}, follow_redirects=True)
        
        # Now, the response status code should be 200 because we are on the final page.
        self.assertEqual(response.status_code, 200)
        
        # Check that the final page contains the prediction result.
        self.assertTrue(
            b'Happy' in response.data or b'Sad' in response.data,
            "Response should contain either 'Happy' or 'Sad'"
        )

if __name__ == '__main__':
    unittest.main()
