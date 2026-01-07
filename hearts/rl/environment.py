import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from game.hearts import HeartsGame
from game.card import Card, Suit, Rank


class RandomBot:
    """Simple random opponent."""
    def select_card(self, legal_moves: List[Card], state: Dict = None) -> Card:
        import random
        return random.choice(legal_moves)



class SmartBot:
    """Expert rule-based opponent with comprehensive Hearts strategy."""
    
    def __init__(self):
        self.played_cards = set()  # Track cards played this round
        self.hearts_broken = False
    
    def reset(self):
        self.played_cards = set()
        self.hearts_broken = False
    
    def select_card(self, legal_moves: List[Card], state: Dict) -> Card:
        if not legal_moves:
            raise ValueError("No legal moves")
        
        # Update tracking from state
        if 'cards_played' in state:
            self.played_cards = set(state['cards_played'])
        if 'hearts_broken' in state:
            self.hearts_broken = state['hearts_broken']
        
        current_trick = state.get('current_trick', [])
        is_lead = state.get('is_lead', len(current_trick) == 0)
        lead_suit = state.get('lead_suit')
        
        # Identify key cards
        qos = Card(Suit.SPADES, Rank.QUEEN)  # Queen of Spades
        aos = Card(Suit.SPADES, Rank.ACE)  # Ace of Spades  
        kos = Card(Suit.SPADES, Rank.KING)  # King of Spades
        
        # Check if QoS has been played
        qos_played = qos in self.played_cards
        aos_played = aos in self.played_cards
        kos_played = kos in self.played_cards
        
        # Categorize legal moves
        hearts = [c for c in legal_moves if c.suit == Suit.HEARTS]
        spades = [c for c in legal_moves if c.suit == Suit.SPADES]
        safe_cards = [c for c in legal_moves if c.penalty_points() == 0]
        
        # ============ LEADING ============
        if is_lead:
            # Strategy: Lead low to avoid winning tricks with points
            
            # If hearts not broken, prefer non-hearts
            if not self.hearts_broken and safe_cards:
                # Lead lowest non-heart, non-spade if QoS still out
                if not qos_played:
                    non_spade = [c for c in safe_cards if c.suit != Suit.SPADES]
                    if non_spade:
                        return min(non_spade, key=lambda c: c.rank)
                return min(safe_cards, key=lambda c: c.rank)
            
            # If hearts broken or only hearts left
            return min(legal_moves, key=lambda c: (c.penalty_points(), c.rank))
        
        # ============ FOLLOWING ============
        following = [c for c in legal_moves if c.suit == lead_suit]
        
        if following:
            # Must follow suit
            trick_cards = [c for _, c in current_trick]
            highest_in_trick = max((c.rank for c in trick_cards if c.suit == lead_suit), default=-1)
            
            # Check if trick has points
            trick_points = sum(c.penalty_points() for _, c in current_trick)
            
            # Strategy: Play highest card that loses, or lowest if must win
            under_cards = [c for c in following if c.rank < highest_in_trick]
            
            if under_cards:
                # Can duck - play highest card that still loses
                return max(under_cards, key=lambda c: c.rank)
            else:
                # Must win - minimize damage
                if trick_points > 0:
                    # Gonna take points, might as well play lowest
                    return min(following, key=lambda c: c.rank)
                else:
                    # Safe to win, play lowest
                    return min(following, key=lambda c: c.rank)
        
        # ============ VOID (CAN'T FOLLOW SUIT) ============
        # Perfect opportunity to dump bad cards!
        
        # Priority 1: Dump Queen of Spades
        if qos in legal_moves:
            return qos
        
        # Priority 2: Dump high spades if QoS still out (to avoid catching it)
        if not qos_played:
            danger_spades = [c for c in spades if c.rank >= 12]  # Q, K, A
            if danger_spades:
                return max(danger_spades, key=lambda c: c.rank)
        
        # Priority 3: Dump high hearts
        if hearts:
            return max(hearts, key=lambda c: c.rank)
        
        # Priority 4: Dump any high spades
        if spades:
            return max(spades, key=lambda c: c.rank)
        
        # Priority 5: Dump highest card overall
        return max(legal_moves, key=lambda c: c.rank)



class HeartsEnv(gym.Env):
    """Hearts environment for RL training."""
    
    def __init__(self, agent_player_id: int = 0, smart_opponents: bool = True, opponents: Optional[List[Any]] = None):
        super().__init__()
        
        self.agent_id = agent_player_id
        self.game = HeartsGame()
        self.smart_opponents = smart_opponents
        
        if opponents is not None:
            self.opponents = opponents
        else:
            self.opponents = [SmartBot() if smart_opponents else RandomBot() for _ in range(3)]
        
        # Action space: 52 cards (not all legal at any time)
        self.action_space = spaces.Discrete(52)
        
        # Observation space
        # - 52: cards in hand (one-hot)
        # - 52: cards played this round (one-hot)
        # - 52: cards in current trick (one-hot)
        # - 4: current trick card count
        # - 4: which player led
        # - 1: hearts broken
        # - 1: first trick
        # - 13: trick number (one-hot)
        # - 4: my points (buckets: 0, 1-6, 7-13, 14+)
        # Total: 183
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(183,), dtype=np.float32
        )
        
        self.prev_my_points = 0
        self.prev_opp_points = [0] * 3
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.game.reset()
        self.prev_my_points = 0
        self.prev_opp_points = [0] * 3
        
        # Play until it's agent's turn
        self._play_until_agent_turn()
        
        obs = self._get_obs(self.agent_id)
        info = self._get_info()
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action (play a card)."""
        # Convert action to card
        card = Card.from_index(action)
        legal_moves = self.game.get_legal_moves(self.agent_id)
        
        # Check if action is legal
        if card not in legal_moves:
            return self._get_obs(self.agent_id), -0.1, False, False, self._get_info()
        
        # Play the card
        reward = 0.0
        trick_winner = self.game.play_card(self.agent_id, card)
        
        # Track points for shaping
        if trick_winner is not None:
            my_new_points = self.game.players[self.agent_id].get_points()
            points_gained = my_new_points - self.prev_my_points
            self.prev_my_points = my_new_points
            
            if points_gained > 0:
                reward -= points_gained * 1.0  # Moderate penalty
            
            # Opp points gained (total for all opps)
            opp_points = [p.get_points() for i, p in enumerate(self.game.players) if i != self.agent_id]
            opp_gained = sum(opp_points) - sum(self.prev_opp_points)
            self.prev_opp_points = opp_points
            if opp_gained > 0:
                reward += opp_gained * 0.5  # Bonus for dumping
            
            # Dump bonus
            if card.penalty_points() > 0 and len(self.game.current_trick) > 0:  # Void dump
                reward += 0.2 * card.penalty_points()
        
        # Early safe play
        if self.game.trick_number < 4 and card.penalty_points() == 0:
            reward += 0.02  # Reduced from 0.1 to prevent conservatism
        
        # Remove noisy exploration bonus - let policy gradient handle it
        
        # Moon pursuit
        if self.game.trick_number > 10 and self.prev_my_points >= 20:
            reward += 1.0
        
        # Let opponents play
        self._play_until_agent_turn()
        
        # Check if round/game over
        done = False
        if self.game.trick_number == 13 or self.game.game_over:
            my_round_score = self.game.round_scores[self.agent_id]
            
            # Shot the moon bonus
            if my_round_score == 26:
                reward = 2.0
            
            # Game over - main reward signal
            if self.game.game_over:
                done = True
                scores = self.game.total_scores
                my_score = scores[self.agent_id]
                
                # Calculate ranking (0 = best, 3 = worst)
                rank = sum(1 for s in scores if s < my_score)
                
                # Winner Takes All Reward Structure
                if rank == 0:
                    reward = 20.0  # Huge bonus for winning
                else:
                    reward = -2.0  # Penalty for losing
            else:
                # Start new round
                self.game.new_round()
                self.prev_my_points = 0
                self.prev_opp_points = [0] * 3
                self._play_until_agent_turn()
        
        obs = self._get_obs(self.agent_id)
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _play_until_agent_turn(self) -> None:
        """Let opponents play until it's the agent's turn."""
        while (self.game.current_player != self.agent_id and 
               self.game.trick_number < 13 and 
               not self.game.game_over):
            
            opp_id = self.game.current_player
            opp_idx = (opp_id - 1) % len(self.opponents)
            
            legal = self.game.get_legal_moves(opp_id)
            if not legal:
                break
            
            if hasattr(self.opponents[opp_idx], 'select_action'):
                # Neural Opponent (PPOAgent)
                obs = self._get_obs(opp_id)
                mask = self._get_mask(opp_id)
                action, _, _ = self.opponents[opp_idx].select_action(obs, mask)
                card = Card.from_index(action)
            elif self.smart_opponents:
                state = self.game.get_state(opp_id)
                card = self.opponents[opp_idx].select_card(legal, state)
            else:
                card = self.opponents[opp_idx].select_card(legal)
            
            self.game.play_card(opp_id, card)
    
    def _get_obs(self, player_id: int) -> np.ndarray:
        """Get observation vector for a specific player."""
        obs = np.zeros(183, dtype=np.float32)
        idx = 0
        
        # Cards in hand (52)
        for card in self.game.players[player_id].hand:
            obs[card.to_index()] = 1.0
        idx += 52
        
        # Cards played this round (52)
        for card in self.game._get_cards_played():
            obs[idx + card.to_index()] = 1.0
        idx += 52
        
        # Cards in current trick (52)
        for _, card in self.game.current_trick:
            obs[idx + card.to_index()] = 1.0
        idx += 52
        
        # Trick card count (4)
        if len(self.game.current_trick) < 4:
            obs[idx + len(self.game.current_trick)] = 1.0
        idx += 4
        
        # Lead player (4)
        obs[idx + self.game.lead_player] = 1.0
        idx += 4
        
        # Hearts broken (1)
        obs[idx] = 1.0 if self.game.hearts_broken else 0.0
        idx += 1
        
        # First trick (1)
        obs[idx] = 1.0 if self.game.first_trick else 0.0
        idx += 1
        
        # Trick number (13)
        if self.game.trick_number < 13:
            obs[idx + self.game.trick_number] = 1.0
        idx += 13
        
        # My points bucket (4)
        pts = self.game.players[player_id].get_points()
        if pts == 0:
            obs[idx] = 1.0
        elif pts <= 6:
            obs[idx + 1] = 1.0
        elif pts <= 13:
            obs[idx + 2] = 1.0
        else:
            obs[idx + 3] = 1.0
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        legal = self.game.get_legal_moves(self.agent_id)
        mask = np.zeros(52, dtype=np.float32)
        for card in legal:
            mask[card.to_index()] = 1.0
        
        winner = self.game.get_winner()
        lead_suit = self.game.current_trick[0][1].suit if self.game.current_trick else None
        
        return {
            'legal_moves': legal,
            'action_mask': mask,
            'trick_number': self.game.trick_number,
            'my_points': self.game.players[self.agent_id].get_points(),
            'winner': winner if winner is not None else -1,
            'is_lead': len(self.game.current_trick) == 0,
            'lead_suit': lead_suit.value if lead_suit else -1,
            'current_trick': self.game.current_trick.copy(),
            'cards_played': self.game._get_cards_played(),
            'hearts_broken': self.game.hearts_broken
        }
    
    def _get_mask(self, player_id: int) -> np.ndarray:
        """Get action mask for a specific player."""
        legal = self.game.get_legal_moves(player_id)
        mask = np.zeros(52, dtype=np.float32)
        for card in legal:
            mask[card.to_index()] = 1.0
        return mask
    
    def action_masks(self) -> np.ndarray:
        """Return action mask for legal moves."""
        return self._get_info()['action_mask']
