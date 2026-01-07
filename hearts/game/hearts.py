import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from .card import Card, Suit, Rank, create_deck


@dataclass
class Player:
    """Represents a player in Hearts."""
    id: int
    hand: List[Card] = field(default_factory=list)
    taken_cards: List[Card] = field(default_factory=list)
    
    def has_suit(self, suit: Suit) -> bool:
        return any(c.suit == suit for c in self.hand)
    
    def has_only_hearts(self) -> bool:
        return all(c.suit == Suit.HEARTS for c in self.hand)
    
    def get_points(self) -> int:
        return sum(c.penalty_points() for c in self.taken_cards)
    
    def remove_card(self, card: Card) -> None:
        self.hand.remove(card)
    
    def add_taken(self, cards: List[Card]) -> None:
        self.taken_cards.extend(cards)


class HeartsGame:
    """Hearts card game implementation."""
    
    def __init__(self):
        self.players: List[Player] = [Player(i) for i in range(4)]
        self.current_player: int = 0
        self.lead_player: int = 0
        self.current_trick: List[Tuple[int, Card]] = []
        self.trick_number: int = 0
        self.hearts_broken: bool = False
        self.first_trick: bool = True
        self.game_over: bool = False
        self.round_scores: List[int] = [0, 0, 0, 0]
        self.total_scores: List[int] = [0, 0, 0, 0]
        
    def reset(self) -> None:
        """Reset for a completely new game."""
        for p in self.players:
            p.hand.clear()
            p.taken_cards.clear()
        
        self.current_trick.clear()
        self.trick_number = 0
        self.hearts_broken = False
        self.first_trick = True
        self.game_over = False
        self.round_scores = [0, 0, 0, 0]
        self.total_scores = [0, 0, 0, 0]  # Reset total scores for new game
        
        # Deal cards
        deck = create_deck()
        random.shuffle(deck)
        for i, card in enumerate(deck):
            self.players[i % 4].hand.append(card)
        
        # Sort hands
        for p in self.players:
            p.hand.sort(key=lambda c: (c.suit, c.rank))
        
        # Find who has 2 of clubs
        two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
        for i, p in enumerate(self.players):
            if two_of_clubs in p.hand:
                self.current_player = i
                self.lead_player = i
                break
    
    def new_round(self) -> None:
        """Start a new round (keep total scores)."""
        for p in self.players:
            p.hand.clear()
            p.taken_cards.clear()
        
        self.current_trick.clear()
        self.trick_number = 0
        self.hearts_broken = False
        self.first_trick = True
        self.round_scores = [0, 0, 0, 0]
        
        # Deal cards
        deck = create_deck()
        random.shuffle(deck)
        for i, card in enumerate(deck):
            self.players[i % 4].hand.append(card)
        
        # Sort hands
        for p in self.players:
            p.hand.sort(key=lambda c: (c.suit, c.rank))
        
        # Find who has 2 of clubs
        two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
        for i, p in enumerate(self.players):
            if two_of_clubs in p.hand:
                self.current_player = i
                self.lead_player = i
                break
    
    def get_legal_moves(self, player_id: int) -> List[Card]:
        """Get legal moves for a player."""
        hand = self.players[player_id].hand
        if not hand:
            return []
        
        # First card of first trick must be 2 of clubs
        if self.first_trick and not self.current_trick:
            two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
            if two_of_clubs in hand:
                return [two_of_clubs]
        
        # If leading
        if not self.current_trick:
            # Can't lead hearts until broken (unless only hearts)
            if not self.hearts_broken:
                non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
                if non_hearts:
                    return non_hearts
            return hand.copy()
        
        # Must follow suit if possible
        lead_suit = self.current_trick[0][1].suit
        same_suit = [c for c in hand if c.suit == lead_suit]
        if same_suit:
            return same_suit
        
        # Can't play hearts or QoS on first trick (unless only have those)
        if self.first_trick:
            safe_cards = [c for c in hand if c.penalty_points() == 0]
            if safe_cards:
                return safe_cards
        
        return hand.copy()
    
    def play_card(self, player_id: int, card: Card) -> Optional[int]:
        """
        Play a card. Returns trick winner if trick complete, None otherwise.
        """
        if player_id != self.current_player:
            raise ValueError(f"Not player {player_id}'s turn")
        
        if card not in self.get_legal_moves(player_id):
            raise ValueError(f"Illegal move: {card}")
        
        # Remove from hand and add to trick
        self.players[player_id].remove_card(card)
        self.current_trick.append((player_id, card))
        
        # Check if hearts broken
        if card.suit == Suit.HEARTS:
            self.hearts_broken = True
        
        # Check if trick complete
        if len(self.current_trick) == 4:
            winner = self._determine_trick_winner()
            trick_cards = [c for _, c in self.current_trick]
            self.players[winner].add_taken(trick_cards)
            
            self.current_trick.clear()
            self.trick_number += 1
            self.first_trick = False
            self.lead_player = winner
            self.current_player = winner
            
            # Check if round over
            if self.trick_number == 13:
                self._end_round()
            
            return winner
        else:
            # Next player
            self.current_player = (self.current_player + 1) % 4
            return None
    
    def _determine_trick_winner(self) -> int:
        """Determine who won the current trick."""
        lead_suit = self.current_trick[0][1].suit
        winner_id = self.current_trick[0][0]
        winner_rank = self.current_trick[0][1].rank
        
        for player_id, card in self.current_trick[1:]:
            if card.suit == lead_suit and card.rank > winner_rank:
                winner_id = player_id
                winner_rank = card.rank
        
        return winner_id
    
    def _end_round(self) -> None:
        """End the current round and calculate scores."""
        # Calculate round scores
        for i, p in enumerate(self.players):
            self.round_scores[i] = p.get_points()
        
        # Check for shooting the moon
        for i in range(4):
            if self.round_scores[i] == 26:
                # Shot the moon! Others get 26 points
                for j in range(4):
                    if j == i:
                        self.round_scores[j] = 0
                    else:
                        self.round_scores[j] = 26
                break
        
        # Add to total scores
        for i in range(4):
            self.total_scores[i] += self.round_scores[i]
        
        # Check if game over (someone reached 100)
        if max(self.total_scores) >= 100:
            self.game_over = True
    
    def get_winner(self) -> Optional[int]:
        """Get the winner (lowest score) if game is over."""
        if not self.game_over:
            return None
        return min(range(4), key=lambda i: self.total_scores[i])
    
    def get_state(self, player_id: int) -> Dict:
        """Get game state from a player's perspective."""
        return {
            'hand': self.players[player_id].hand.copy(),
            'current_trick': self.current_trick.copy(),
            'is_lead': len(self.current_trick) == 0,
            'lead_suit': self.current_trick[0][1].suit if self.current_trick else None,
            'hearts_broken': self.hearts_broken,
            'first_trick': self.first_trick,
            'trick_number': self.trick_number,
            'current_player': self.current_player,
            'legal_moves': self.get_legal_moves(player_id),
            'my_points': self.players[player_id].get_points(),
            'total_scores': self.total_scores.copy(),
            'round_scores': self.round_scores.copy(),
            'cards_played': self._get_cards_played(),
            'game_over': self.game_over
        }
    
    def _get_cards_played(self) -> List[Card]:
        """Get all cards that have been played this round."""
        played = []
        for p in self.players:
            played.extend(p.taken_cards)
        for _, card in self.current_trick:
            played.append(card)
        return played

