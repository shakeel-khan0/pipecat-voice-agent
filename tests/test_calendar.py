import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]

# -------------------------
# GOOGLE AUTHENTICATION
# -------------------------

creds = None

if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file(
        "token.json",
        SCOPES
    )

if not creds or not creds.valid:

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            SCOPES
        )

        creds = flow.run_local_server(port=0)

    with open("token.json", "w") as token:
        token.write(creds.to_json())


# Create Google Calendar service
service = build(
    "calendar",
    "v3",
    credentials=creds
)


# -------------------------
# READ TOMORROW'S EVENTS
# -------------------------

tz = ZoneInfo("Asia/Karachi")

now = datetime.now(tz)

tomorrow = (now + timedelta(days=1)).date()

start = datetime.combine(
    tomorrow,
    datetime.min.time(),
    tzinfo=tz
)

end = start + timedelta(days=1)

events = service.events().list(
    calendarId="primary",
    timeMin=start.isoformat(),
    timeMax=end.isoformat(),
    singleEvents=True,
    orderBy="startTime",
).execute()


print(f"\nEvents on {tomorrow}:")

for event in events.get("items", []):

    print(
        event.get("summary", "No title"),
        "|",
        event["start"].get(
            "dateTime",
            event["start"].get("date")
        ),
        "→",
        event["end"].get(
            "dateTime",
            event["end"].get("date")
        )
    )