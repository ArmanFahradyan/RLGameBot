from enum import IntEnum
from dataclasses import dataclass
from typing import List


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    SPADES = 2
    HEARTS = 3


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


SUIT_NAMES = ['♣', '♦', '♠', '♥']
RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank
    
    def __repr__(self) -> str:
        return f"{RANK_NAMES[self.rank]}{SUIT_NAMES[self.suit]}"
    
    def to_index(self) -> int:
        """Convert card to index 0-51."""
        return self.suit * 13 + self.rank
    
    @staticmethod
    def from_index(idx: int) -> 'Card':
        """Create card from index 0-51."""
        return Card(Suit(idx // 13), Rank(idx % 13))
    
    def penalty_points(self) -> int:
        """Return penalty points for this card."""
        if self.suit == Suit.HEARTS:
            return 1
        if self.suit == Suit.SPADES and self.rank == Rank.QUEEN:
            return 13
        return 0


def create_deck() -> List[Card]:
    """Create a standard 52-card deck."""
    return [Card(suit, rank) for suit in Suit for rank in Rank]

