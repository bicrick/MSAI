from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        # Find min and max along dimension 1 (measurements within each day)
        min_per_day = torch.min(self.data, dim=1).values
        max_per_day = torch.max(self.data, dim=1).values
        return min_per_day, max_per_day

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        # Calculate daily averages
        daily_avgs = torch.mean(self.data, dim=1)
        # Calculate day-over-day differences
        temp_changes = daily_avgs[1:] - daily_avgs[:-1]
        # Find the largest negative change (minimum value)
        return torch.min(temp_changes)

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature.
        """
        # Calculate daily averages (keeping dims for broadcasting)
        daily_avgs = torch.mean(self.data, dim=1, keepdim=True)
        # Compute absolute differences between each measurement and the day's average
        differences = torch.abs(self.data - daily_avgs)
        # For each day, find the index of the measurement with the maximum difference
        indices = torch.argmax(differences, dim=1)
        # Gather the corresponding measurements using the found indices
        most_extreme_measurements = torch.gather(self.data, 1, indices.unsqueeze(1)).squeeze(1)
        return most_extreme_measurements

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        # Get last k days of data
        last_k_data = self.data[-k:]
        # Find max temperature for each day
        return torch.max(last_k_data, dim=1).values

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        # Get last k days of data
        last_k_data = self.data[-k:]
        # Calculate average of all measurements over last k days
        return torch.mean(last_k_data)

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        Find which day in the dataset most closely matches the input measurements
        """
        # Ensure input tensor has shape [10] and data has shape [num_days, 10]
        if t.dim() == 1:
            t = t.view(1, -1)  # reshape to [1, 10]
        
        # Calculate absolute differences between input and all days
        differences = torch.abs(self.data - t)
        # Sum differences for each day to get total difference per day
        total_differences = torch.sum(differences, dim=1)
        # Find day with minimum total difference
        closest_day = torch.argmin(total_differences)
        return closest_day
