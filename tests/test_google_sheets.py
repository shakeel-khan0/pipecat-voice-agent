"""Unit tests for strict Google Sheets append confirmation."""

import unittest
from unittest.mock import MagicMock, patch

from services import google_sheets


class GoogleSheetsTests(unittest.TestCase):
    def _service_with_response(self, response):
        service = MagicMock()
        service.spreadsheets.return_value.values.return_value.append.return_value.execute.return_value = response
        return service

    def _save(self):
        return google_sheets.save_lead(
            name="Test Caller",
            email="test@example.com",
            business="",
            industry="",
            problem="",
            volume="",
            current_system="",
            interested_service="",
            meeting_date="2026-09-20",
            meeting_time="04:00 PM",
        )

    def test_success_requires_confirmed_full_row(self):
        response = {
            "updates": {
                "updatedRange": "Sheet1!A7:K7",
                "updatedRows": 1,
                "updatedColumns": 11,
                "updatedCells": 11,
                "updatedData": {"values": [["value"] * 11]},
            }
        }
        with patch.object(google_sheets, "get_sheets_service", return_value=self._service_with_response(response)):
            result = self._save()

        self.assertTrue(result["success"])
        self.assertEqual(result["updated_range"], "Sheet1!A7:K7")
        self.assertEqual(result["updated_rows"], 1)

    def test_missing_api_write_confirmation_returns_failure(self):
        response = {"updates": {"updatedRange": "Sheet1!A7:K7"}}
        with patch.object(google_sheets, "get_sheets_service", return_value=self._service_with_response(response)):
            result = self._save()

        self.assertFalse(result["success"])
        self.assertIn("did not confirm", result["error"])

    def test_payload_maps_eleven_columns_to_target_range(self):
        response = {
            "updates": {
                "updatedRange": "Sheet1!A7:K7",
                "updatedRows": 1,
                "updatedColumns": 11,
                "updatedCells": 11,
                "updatedData": {"values": [["value"] * 11]},
            }
        }
        service = self._service_with_response(response)
        with patch.object(google_sheets, "get_sheets_service", return_value=service):
            self._save()

        append = service.spreadsheets.return_value.values.return_value.append
        kwargs = append.call_args.kwargs
        self.assertEqual(kwargs["range"], "Sheet1!A:K")
        self.assertTrue(kwargs["includeValuesInResponse"])
        values = kwargs["body"]["values"][0]
        self.assertEqual(len(values), 11)
        self.assertEqual(values[1], "Test Caller")
        self.assertEqual(values[2], "test@example.com")
        self.assertEqual(values[9], "2026-09-20")
        self.assertEqual(values[10], "04:00 PM")


if __name__ == "__main__":
    unittest.main()
