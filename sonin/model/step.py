class HasStep:
    def step(self, c_time: int):
        """
        Advance time forward one step.
        `c_time` is a monotonically incrementing step number.
        """

        pass

    def cleanup(self, c_time: int):
        """
        Cleans up state to prepare for the next step.
        """

        pass
