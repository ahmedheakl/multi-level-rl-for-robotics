class Doxy:

    """Main class for generating doxy envs"""

    def __init__(self, id: int = 3) -> None:
        """Create main id data

        Args:
            id (int, optional): id of current user. Defaults to 3.
        """
        self.id = id

    def doxy(self, name: str = "hamed") -> str:
        """Returns the same name with 1 added

        Args:
            name (str, optional): name of the current user. Defaults to "hamed".

        Returns:
            str: output result for user
        """
        return name + "1"
