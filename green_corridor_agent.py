import os
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, WebSocket, BackgroundTasks
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GreenCorridorAgent")

# --- Data Models (Pydantic) ---

class GeoPoint(BaseModel):
    lat: float
    lon: float

class TrafficStatus(str, Enum):
    CLEAR = "CLEAR"
    MODERATE = "MODERATE"
    CONGESTED = "CONGESTED"
    GRIDLOCK = "GRIDLOCK"

class RouteSegment(BaseModel):
    segment_id: str
    start: GeoPoint
    end: GeoPoint
    traffic_status: TrafficStatus = TrafficStatus.CLEAR
    is_green_corridor_active: bool = False

class AmbulanceTelemetry(BaseModel):
    ambulance_id: str
    location: GeoPoint
    speed_kmh: float
    heading: float
    timestamp: datetime = Field(default_factory=datetime.now)

# --- The Green Corridor Agent ---

class GreenCorridorAgent:
    """
    Autonomous agent managing the Green Corridor for an ambulance.
    Leverages Gemini 2.5 Flash for decision making based on complex traffic data.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.active_route: List[RouteSegment] = []
        self.ambulance_status: Optional[AmbulanceTelemetry] = None
        self.model_name = model_name
        self.client = client
        
        logger.info(f"Agent initialized with {model_name}")

    # --- Deterministic Tool Implementations ---

    def create_green_corridor(self, segment_ids: List[str]):
        """
        Highlights the path on the map and sets traffic signals to Green.
        """
        logger.info(f"ACTIVATING GREEN CORRIDOR for segments: {segment_ids}")
        for seg in self.active_route:
            if seg.segment_id in segment_ids:
                seg.is_green_corridor_active = True
                print(f"[System] Traffic Lights set to GREEN for segment {seg.segment_id}")
        return {"status": "success", "active_segments": segment_ids}

    def notify_vehicles(self, segment_id: str, message: str):
        """
        Broadcasts alerts to civilian vehicles on the path.
        """
        logger.info(f"BROADCASTING to vehicles on {segment_id}: {message}")
        return {"status": "sent", "recipient_count": 150}

    def notify_police(self, location: str, urgency: str):
        """
        Notifies traffic police control room.
        """
        logger.warning(f"POLICE ALERT [{urgency}]: Assistance needed at {location}")
        return {"status": "acknowledged", "unit_dispatched": True}

    def auto_update_route_if_needed(self, current_segment_id: str, traffic_density: float):
        """
        Recalculates traffic. If density > 0.8 (80%), triggers rerouting.
        """
        if traffic_density > 0.8:
            logger.info(f"Traffic critical on {current_segment_id}. Recalculating...")
            new_route_id = "route_alt_2B"
            return {"action": "rerouted", "new_route_id": new_route_id}
        return {"action": "maintain_course", "status": "optimal"}

    # --- Core Loop & LLM Interaction ---

    async def process_telemetry(self, telemetry: AmbulanceTelemetry, traffic_data: Dict[str, float]):
        """
        The 'Brain' of the agent. Sends current state to Gemini 2.5 Flash 
        to decide on the next best action.
        """
        self.ambulance_status = telemetry
        
        # Construct context for the LLM
        prompt = f"""
        You are managing a Green Corridor for an ambulance. Analyze the situation and decide actions.
        
        Ambulance ID: {telemetry.ambulance_id}
        Location: {telemetry.location.lat}, {telemetry.location.lon}
        Speed: {telemetry.speed_kmh} km/h
        
        Current Traffic Conditions (Density 0-1):
        {traffic_data}
        
        Protocol:
        - If traffic is high (>0.7), recommend warning police.
        - If traffic is extreme (>0.8), recommend rerouting.
        - Identify which segments need green corridor activation.
        
        Provide your analysis and recommended actions.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            decision_text = response.text
            logger.info(f"Agent Decision Log: {decision_text}")
            
            # Parse LLM response and trigger actions
            # For demo purposes, we'll trigger based on traffic data directly
            high_traffic_segments = [seg_id for seg_id, density in traffic_data.items() if density > 0.7]
            
            if high_traffic_segments:
                for seg_id in high_traffic_segments:
                    if traffic_data[seg_id] > 0.8:
                        self.auto_update_route_if_needed(seg_id, traffic_data[seg_id])
                        self.notify_police(f"Segment {seg_id}", "Critical")
                    else:
                        self.notify_police(f"Segment {seg_id}", "High")
            
            # Activate green corridor for next segments
            upcoming_segments = ["A1", "A2"]  # In real system, calculate based on ambulance position
            self.create_green_corridor(upcoming_segments)
            
            return decision_text
            
        except Exception as e:
            logger.error(f"LLM Inference failed: {e}")
            return "Error in agent decision logic."

    def track_ambulance_progress(self, step_index: int):
        """
        Updates the UI on progress. Clears the route behind the ambulance.
        """
        if 0 <= step_index < len(self.active_route):
            completed_seg = self.active_route[step_index]
            completed_seg.is_green_corridor_active = False
            print(f"[System] Clearing Green Corridor for passed segment: {completed_seg.segment_id}")

# --- FastAPI Interface ---

app = FastAPI(title="Green Corridor AI Agent")
agent = GreenCorridorAgent()

# Mock Database of Route
agent.active_route = [
    RouteSegment(segment_id="A1", start=GeoPoint(lat=12.9, lon=77.5), end=GeoPoint(lat=12.91, lon=77.51)),
    RouteSegment(segment_id="A2", start=GeoPoint(lat=12.91, lon=77.51), end=GeoPoint(lat=12.92, lon=77.52)),
    RouteSegment(segment_id="A3", start=GeoPoint(lat=12.92, lon=77.52), end=GeoPoint(lat=12.93, lon=77.53)),
]

@app.post("/telemetry/update")
async def update_telemetry(data: AmbulanceTelemetry, background_tasks: BackgroundTasks):
    """
    Receives GPS updates from the ambulance every second.
    """
    current_traffic = {"A1": 0.1, "A2": 0.4, "A3": 0.85}
    background_tasks.add_task(agent.process_telemetry, data, current_traffic)
    return {"status": "processing", "timestamp": datetime.now()}

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """
    Continuous UI updates for the Command Center.
    """
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "ambulance": agent.ambulance_status.dict() if agent.ambulance_status else None,
                "route_segments": [s.dict() for s in agent.active_route]
            })
            await asyncio.sleep(1)
    except Exception:
        print("Client disconnected")

# --- Usage Example ---

if __name__ == "__main__":
    print("--- Starting Simulation ---")
    
    tel = AmbulanceTelemetry(
        ambulance_id="AMB-99", 
        location=GeoPoint(lat=12.91, lon=77.51), 
        speed_kmh=45.0, 
        heading=90.0
    )
    
    print(f"Processing telemetry for {tel.ambulance_id}...")
    
    traffic_snapshot = {"A1": 0.1, "A2": 0.4, "A3": 0.85}
    
    agent.create_green_corridor(["A2"])
    result = agent.auto_update_route_if_needed("A3", 0.85)
    print(f"Agent Action Result: {result}")
    agent.notify_police("Sector A3", "High Congestion [0.85]")
    
    print("--- Simulation End ---")
