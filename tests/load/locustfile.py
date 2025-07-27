"""Load testing configuration using Locust."""

import json
import random
from locust import HttpUser, task, between


class CausalEvalUser(HttpUser):
    """Simulate a user of the Causal Eval Bench API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup that runs when a user starts."""
        # Simulate user login/authentication if needed
        self.client.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "LoadTest/1.0"
        })
    
    @task(3)
    def health_check(self):
        """Test health endpoint (frequent check)."""
        self.client.get("/health")
    
    @task(5)
    def get_questions(self):
        """Get evaluation questions."""
        params = {
            "task_type": random.choice(["causal_attribution", "counterfactual", "intervention"]),
            "domain": random.choice(["medical", "social", "economic", "scientific"]),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "limit": random.randint(5, 20)
        }
        
        response = self.client.get("/api/v1/questions", params=params)
        
        if response.status_code == 200:
            questions = response.json()
            # Store questions for evaluation
            if questions and "questions" in questions:
                self.questions = questions["questions"][:5]  # Take first 5
    
    @task(8)
    def evaluate_model(self):
        """Submit model evaluation."""
        if not hasattr(self, 'questions') or not self.questions:
            # Get some questions first
            self.get_questions()
            return
        
        # Simulate model evaluation
        evaluation_data = {
            "model_name": f"TestModel-{random.randint(1, 100)}",
            "questions": self.questions,
            "responses": [
                {
                    "question_id": q["id"],
                    "response": random.choice([
                        "Yes, there is a causal relationship.",
                        "No, this is just correlation.",
                        "The relationship is unclear."
                    ]),
                    "confidence": random.uniform(0.1, 1.0)
                }
                for q in self.questions
            ]
        }
        
        response = self.client.post(
            "/api/v1/evaluate",
            data=json.dumps(evaluation_data)
        )
        
        if response.status_code == 200:
            evaluation_result = response.json()
            if "evaluation_id" in evaluation_result:
                self.evaluation_id = evaluation_result["evaluation_id"]
    
    @task(2)
    def get_evaluation_result(self):
        """Check evaluation results."""
        if hasattr(self, 'evaluation_id'):
            self.client.get(f"/api/v1/evaluations/{self.evaluation_id}")
    
    @task(1)
    def get_leaderboard(self):
        """Check leaderboard."""
        self.client.get("/api/v1/leaderboard")
    
    @task(2)
    def generate_questions(self):
        """Generate new questions."""
        generation_data = {
            "task_type": random.choice(["causal_attribution", "counterfactual"]),
            "domain": random.choice(["medical", "social", "economic"]),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "count": random.randint(1, 5),
            "seed": random.randint(1, 1000)
        }
        
        self.client.post(
            "/api/v1/questions/generate",
            data=json.dumps(generation_data)
        )
    
    @task(1)
    def get_metrics(self):
        """Get system metrics."""
        self.client.get("/metrics")


class AdminUser(HttpUser):
    """Simulate admin user performing administrative tasks."""
    
    wait_time = between(5, 10)  # Admins are less frequent
    weight = 1  # Fewer admin users
    
    def on_start(self):
        """Admin setup."""
        self.client.headers.update({
            "Content-Type": "application/json",
            "Authorization": "Bearer admin-token",  # Simulated admin token
            "User-Agent": "AdminLoadTest/1.0"
        })
    
    @task(3)
    def view_admin_dashboard(self):
        """View admin dashboard."""
        self.client.get("/admin/dashboard")
    
    @task(2)
    def manage_questions(self):
        """Manage question database."""
        # Get questions
        self.client.get("/admin/questions")
        
        # Add a question
        new_question = {
            "prompt": f"Admin test question {random.randint(1, 1000)}",
            "task_type": "causal_attribution",
            "domain": "test",
            "difficulty": "medium",
            "ground_truth": {
                "answer": "Test answer",
                "explanation": "Test explanation"
            }
        }
        
        self.client.post(
            "/admin/questions",
            data=json.dumps(new_question)
        )
    
    @task(1)
    def view_system_stats(self):
        """View system statistics."""
        self.client.get("/admin/stats")
    
    @task(1)
    def export_data(self):
        """Export evaluation data."""
        export_params = {
            "format": random.choice(["json", "csv"]),
            "date_from": "2024-01-01",
            "date_to": "2024-12-31"
        }
        
        self.client.get("/admin/export", params=export_params)


class HeavyUser(HttpUser):
    """Simulate heavy usage scenarios."""
    
    wait_time = between(0.5, 2)  # More frequent requests
    weight = 2  # More heavy users
    
    @task(10)
    def batch_evaluation(self):
        """Perform batch evaluation."""
        # Generate large batch of questions
        batch_data = {
            "model_name": f"HeavyTestModel-{random.randint(1, 50)}",
            "batch_size": random.randint(50, 100),
            "task_types": ["causal_attribution", "counterfactual", "intervention"],
            "domains": ["medical", "economic", "social"],
            "difficulty_levels": ["easy", "medium", "hard"]
        }
        
        self.client.post(
            "/api/v1/evaluate/batch",
            data=json.dumps(batch_data)
        )
    
    @task(5)
    def stream_evaluations(self):
        """Test streaming evaluation endpoint."""
        stream_data = {
            "model_name": f"StreamModel-{random.randint(1, 20)}",
            "stream": True,
            "batch_size": random.randint(10, 30)
        }
        
        with self.client.post(
            "/api/v1/evaluate/stream",
            data=json.dumps(stream_data),
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Streaming failed: {response.status_code}")


# Custom load testing scenarios
class SpikeTestUser(HttpUser):
    """Simulate traffic spikes."""
    
    wait_time = between(0.1, 0.5)  # Very fast requests during spike
    
    @task
    def rapid_fire_requests(self):
        """Send rapid requests to test spike handling."""
        endpoints = [
            "/health",
            "/api/v1/questions",
            "/api/v1/leaderboard",
            "/metrics"
        ]
        
        endpoint = random.choice(endpoints)
        self.client.get(endpoint)


# Configuration for different test scenarios
if __name__ == "__main__":
    # This allows running specific user types
    import sys
    
    if len(sys.argv) > 1:
        user_type = sys.argv[1]
        if user_type == "heavy":
            print("Running heavy user simulation")
        elif user_type == "admin":
            print("Running admin user simulation")
        elif user_type == "spike":
            print("Running spike test simulation")
        else:
            print("Running normal user simulation")
