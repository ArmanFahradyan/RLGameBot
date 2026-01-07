import streamlit as st
import torch
import numpy as np
import os
import sys
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Game Hub",
    page_icon="🎮",
    layout="wide"
)

# Global CSS
st.markdown("""
<style>
    .game-title {
        text-align: center;
        font-size: 48px;
        background: linear-gradient(135deg, #00d4ff 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .last-trick {
        padding: 15px;
        background: #1a1a2e;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
    }
    .trick-winner {
        color: #ffd700;
        font-weight: bold;
        font-size: 18px;
    }
    .card-red {
        color: #ff4757 !important;
        font-weight: bold;
        font-size: 18px;
    }
    .card-black {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 18px;
    }
    .piece-black {
        font-size: 36px;
        text-shadow: 0 0 10px #000000;
    }
    .piece-white {
        font-size: 36px;
        text-shadow: 0 0 10px #ffffff;
    }
    .trick-card {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        background: #16213e;
        border-radius: 5px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ============== ALPHAZERO GAMES ==============
def load_alphazero_model(game_name):
    """Load AlphaZero model for a game."""
    from src.game_factory import get_game_config
    from src.network import PolicyValueNet
    
    config = get_game_config(game_name)
    model_path = config.model_save_path
    
    if not os.path.exists(model_path):
        return None, config
    
    network = PolicyValueNet(
        board_size=config.board_size,
        action_size=config.action_size,
        hidden_size=config.hidden_size
    )
    network.load_state_dict(torch.load(model_path, map_location='cpu'))
    network.eval()
    return network, config


def get_ai_move(game, network, state, config):
    """Get AI move using MCTS."""
    from src.mcts import MCTS, MCTSNode
    
    root = MCTSNode(state, prior=1.0)
    mcts = MCTS(game, network, config.c_puct)
    
    for _ in range(min(config.mcts_simulations, 50)):
        mcts.simulate(root)
    
    best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
    return best_action


# ============== TICTACTOE ==============
def play_tictactoe():
    st.markdown('<div class="game-title">⭕ Tic-Tac-Toe ❌</div>', unsafe_allow_html=True)
    
    from src.game import TicTacToe
    game = TicTacToe()
    
    # Side selection
    if 'ttt_side' not in st.session_state:
        st.markdown("### Choose Your Side")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❌ Play as X (First)", use_container_width=True):
                st.session_state.ttt_side = 1
                st.session_state.ttt_state = game.initial_state()
                st.session_state.ttt_network, st.session_state.ttt_config = load_alphazero_model('tictactoe')
                st.session_state.ttt_message = "Your turn! You are ❌."
                st.rerun()
        with col2:
            if st.button("⭕ Play as O (Second)", use_container_width=True):
                st.session_state.ttt_side = -1
                st.session_state.ttt_state = game.initial_state()
                st.session_state.ttt_network, st.session_state.ttt_config = load_alphazero_model('tictactoe')
                # AI moves first
                network = st.session_state.ttt_network
                config = st.session_state.ttt_config
                if network:
                    ai_action = get_ai_move(game, network, st.session_state.ttt_state, config)
                    st.session_state.ttt_state = game.next_state(st.session_state.ttt_state, ai_action)
                st.session_state.ttt_message = "AI played first. Your turn! You are ⭕."
                st.rerun()
        return
    
    human_side = st.session_state.ttt_side
    state = st.session_state.ttt_state
    board, player = state
    network = st.session_state.ttt_network
    config = st.session_state.ttt_config
    
    if network is None:
        st.error("Model not found! Train first: `python -m src.main` → Select 1")
        if st.button("🔄 Reset"):
            for key in list(st.session_state.keys()):
                if key.startswith('ttt_'):
                    del st.session_state[key]
            st.rerun()
        return
    
    # Check game over
    if game.is_terminal(state):
        result = game.outcome(state)
        if result == human_side:
            st.success("🎉 You Win!")
        elif result == -human_side:
            st.error("🤖 AI Wins!")
        else:
            st.info("🤝 Draw!")
        if st.button("🔄 Play Again"):
            for key in list(st.session_state.keys()):
                if key.startswith('ttt_'):
                    del st.session_state[key]
            st.rerun()
        return
    
    symbols = {1: "❌", -1: "⭕", 0: ""}
    st.info(st.session_state.ttt_message)
    st.write(f"You are: {symbols[human_side]}")
    
    # Display board
    cols = st.columns(3)
    
    for i in range(9):
        col = cols[i % 3]
        with col:
            cell_val = board[i]
            if cell_val == 0 and player == human_side:
                if st.button(symbols[cell_val] or "·", key=f"ttt_{i}", use_container_width=True):
                    if i in game.legal_actions(state):
                        state = game.next_state(state, i)
                        st.session_state.ttt_state = state
                        
                        if not game.is_terminal(state):
                            ai_action = get_ai_move(game, network, state, config)
                            state = game.next_state(state, ai_action)
                            st.session_state.ttt_state = state
                            st.session_state.ttt_message = f"AI played position {ai_action}. Your turn!"
                        st.rerun()
            else:
                st.button(symbols[cell_val] or "·", key=f"ttt_{i}", disabled=True, use_container_width=True)
        
        if (i + 1) % 3 == 0 and i < 8:
            cols = st.columns(3)
    
    if st.button("🔄 Reset Game"):
        for key in list(st.session_state.keys()):
            if key.startswith('ttt_'):
                del st.session_state[key]
        st.rerun()


# ============== CONNECT FOUR ==============
def play_connect_four():
    st.markdown('<div class="game-title">🔴 Connect Four 🟡</div>', unsafe_allow_html=True)
    
    from src.connect_four import ConnectFour
    game = ConnectFour()
    
    # Side selection
    if 'c4_side' not in st.session_state:
        st.markdown("### Choose Your Side")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Play as Red (First)", use_container_width=True):
                st.session_state.c4_side = 1
                st.session_state.c4_state = game.initial_state()
                st.session_state.c4_network, st.session_state.c4_config = load_alphazero_model('connect_four')
                st.session_state.c4_message = "Your turn! You are 🔴."
                st.rerun()
        with col2:
            if st.button("🟡 Play as Yellow (Second)", use_container_width=True):
                st.session_state.c4_side = -1
                st.session_state.c4_state = game.initial_state()
                st.session_state.c4_network, st.session_state.c4_config = load_alphazero_model('connect_four')
                network = st.session_state.c4_network
                config = st.session_state.c4_config
                if network:
                    ai_action = get_ai_move(game, network, st.session_state.c4_state, config)
                    st.session_state.c4_state = game.next_state(st.session_state.c4_state, ai_action)
                st.session_state.c4_message = "AI played first. Your turn! You are 🟡."
                st.rerun()
        return
    
    human_side = st.session_state.c4_side
    state = st.session_state.c4_state
    board, player = state
    network = st.session_state.c4_network
    config = st.session_state.c4_config
    
    if network is None:
        st.error("Model not found! Train first: `python -m src.main` → Select 2")
        if st.button("🔄 Reset"):
            for key in list(st.session_state.keys()):
                if key.startswith('c4_'):
                    del st.session_state[key]
            st.rerun()
        return
    
    if game.is_terminal(state):
        result = game.outcome(state)
        if result == human_side:
            st.success("🎉 You Win!")
        elif result == -human_side:
            st.error("🤖 AI Wins!")
        else:
            st.info("🤝 Draw!")
        if st.button("🔄 Play Again"):
            for key in list(st.session_state.keys()):
                if key.startswith('c4_'):
                    del st.session_state[key]
            st.rerun()
        return
    
    symbols = {1: "🔴", -1: "🟡", 0: "⚪"}
    st.info(st.session_state.c4_message)
    st.write(f"You are: {symbols[human_side]}")
    
    # Column buttons
    cols = st.columns(7)
    legal = game.legal_actions(state)
    
    for col_idx, col in enumerate(cols):
        with col:
            disabled = col_idx not in legal or player != human_side
            if st.button(f"⬇️ {col_idx}", key=f"c4_col_{col_idx}", disabled=disabled, use_container_width=True):
                state = game.next_state(state, col_idx)
                st.session_state.c4_state = state
                
                if not game.is_terminal(state):
                    ai_action = get_ai_move(game, network, state, config)
                    state = game.next_state(state, ai_action)
                    st.session_state.c4_state = state
                    st.session_state.c4_message = f"AI played column {ai_action}. Your turn!"
                st.rerun()
    
    # Display board
    board_2d = board.reshape(6, 7)
    
    for row in range(6):
        cols = st.columns(7)
        for col_idx, col in enumerate(cols):
            with col:
                st.markdown(f"<div style='text-align:center;font-size:32px'>{symbols[board_2d[row, col_idx]]}</div>", unsafe_allow_html=True)
    
    if st.button("🔄 Reset Game"):
        for key in list(st.session_state.keys()):
            if key.startswith('c4_'):
                del st.session_state[key]
        st.rerun()


# ============== OTHELLO ==============
def play_othello():
    st.markdown('<div class="game-title">⚫ Othello ⚪</div>', unsafe_allow_html=True)
    
    from src.othello import Othello
    game = Othello()
    
    # Side selection
    if 'oth_side' not in st.session_state:
        st.markdown("### Choose Your Side")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔵 Play as Blue (First)", use_container_width=True):
                st.session_state.oth_side = 1
                st.session_state.oth_state = game.initial_state()
                st.session_state.oth_network, st.session_state.oth_config = load_alphazero_model('othello')
                st.session_state.oth_message = "Your turn! You are 🔵 Blue."
                st.rerun()
        with col2:
            if st.button("🟠 Play as Orange (Second)", use_container_width=True):
                st.session_state.oth_side = -1
                st.session_state.oth_state = game.initial_state()
                st.session_state.oth_network, st.session_state.oth_config = load_alphazero_model('othello')
                network = st.session_state.oth_network
                config = st.session_state.oth_config
                state = st.session_state.oth_state
                # AI moves first
                if network:
                    ai_action = get_ai_move(game, network, state, config)
                else:
                    ai_action = np.random.choice(game.legal_actions(state))
                st.session_state.oth_state = game.next_state(state, ai_action)
                st.session_state.oth_message = "AI played first. Your turn! You are 🟠 Orange."
                st.rerun()
        return

    
    human_side = st.session_state.oth_side
    state = st.session_state.oth_state
    board, player = state
    network = st.session_state.oth_network
    config = st.session_state.oth_config
    
    if network is None:
        st.warning("Model not found! Using random AI.")
    
    if game.is_terminal(state):
        black, white = game.get_score(state)
        if (black > white and human_side == 1) or (white > black and human_side == -1):
            st.success(f"🎉 You Win! ({black} - {white})")
        elif (white > black and human_side == 1) or (black > white and human_side == -1):
            st.error(f"🤖 AI Wins! ({black} - {white})")
        else:
            st.info(f"🤝 Draw! ({black} - {white})")
        if st.button("🔄 Play Again"):
            for key in list(st.session_state.keys()):
                if key.startswith('oth_'):
                    del st.session_state[key]
            st.rerun()
        return
    
    # Use high-contrast colors for Othello
    symbols = {1: "🔵", -1: "🟠", 0: ""}
    symbol_names = {1: "🔵 Blue", -1: "🟠 Orange"}
    st.info(st.session_state.oth_message)
    st.write(f"You are: {symbol_names[human_side]}")
    
    # Score
    black, white = game.get_score(state)
    st.markdown(f"**Score:** 🔵 Blue: {black} | 🟠 Orange: {white}")

    
    legal = game.legal_actions(state)
    board_2d = board.reshape(8, 8)
    
    # Display board
    for row in range(8):
        cols = st.columns(8)
        for col_idx, col in enumerate(cols):
            with col:
                pos = row * 8 + col_idx
                cell = board_2d[row, col_idx]
                
                if pos in legal and player == human_side:
                    if st.button("🟢", key=f"oth_{pos}", use_container_width=True):
                        state = game.next_state(state, pos)
                        st.session_state.oth_state = state
                        
                        # AI move(s)
                        while not game.is_terminal(state) and state[1] == -human_side:
                            ai_legal = game.legal_actions(state)
                            if network:
                                ai_action = get_ai_move(game, network, state, config)
                            else:
                                ai_action = np.random.choice(ai_legal)
                            state = game.next_state(state, ai_action)
                            st.session_state.oth_state = state
                            if ai_action == 64:
                                st.session_state.oth_message = "AI passed. Your turn!"
                            else:
                                st.session_state.oth_message = f"AI played {ai_action}. Your turn!"
                        st.rerun()
                elif cell != 0:
                    st.button(symbols[cell], key=f"oth_{pos}", disabled=True, use_container_width=True)
                else:
                    st.button("·", key=f"oth_{pos}", disabled=True, use_container_width=True)
    
    # Handle pass
    if legal == [64] and player == human_side:
        st.warning("No legal moves! You must pass.")
        if st.button("Pass"):
            state = game.next_state(state, 64)
            st.session_state.oth_state = state
            st.rerun()
    
    if st.button("🔄 Reset Game"):
        for key in list(st.session_state.keys()):
            if key.startswith('oth_'):
                del st.session_state[key]
        st.rerun()


# ============== HEARTS ==============
def play_hearts():
    st.markdown('<div class="game-title">♥️ Hearts ♥️</div>', unsafe_allow_html=True)
    
    # Import Hearts modules
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hearts'))
        from hearts.game.hearts import HeartsGame
        from hearts.game.card import Card, Suit, Rank, SUIT_NAMES, RANK_NAMES
        from hearts.rl.agent import PPOAgent
        from hearts.rl.environment import SmartBot, RandomBot
    except ImportError as e:
        st.error(f"Could not import Hearts modules: {e}")
        return
    
    def get_obs(game, player_id):
        obs = np.zeros(183, dtype=np.float32)
        for card in game.players[player_id].hand:
            obs[card.to_index()] = 1.0
        idx = 52
        for card in game._get_cards_played():
            obs[idx + card.to_index()] = 1.0
        idx += 52
        for _, card in game.current_trick:
            obs[idx + card.to_index()] = 1.0
        idx += 52
        if len(game.current_trick) < 4:
            obs[idx + len(game.current_trick)] = 1.0
        idx += 4
        obs[idx + game.lead_player] = 1.0
        idx += 4
        obs[idx] = 1.0 if game.hearts_broken else 0.0
        idx += 1
        obs[idx] = 1.0 if game.first_trick else 0.0
        idx += 1
        if game.trick_number < 13:
            obs[idx + game.trick_number] = 1.0
        idx += 13
        pts = game.players[player_id].get_points()
        if pts == 0:
            obs[idx] = 1.0
        elif pts <= 6:
            obs[idx + 1] = 1.0
        elif pts <= 13:
            obs[idx + 2] = 1.0
        else:
            obs[idx + 3] = 1.0
        return obs
    
    def get_mask(game, player_id):
        mask = np.zeros(52, dtype=np.float32)
        for card in game.get_legal_moves(player_id):
            mask[card.to_index()] = 1.0
        return mask
    
    def load_hearts_agent():
        paths = [
            'models/hearts_rl_agent.pt',
            'checkpoints/hearts_rl_agent.pt',
            'hearts/checkpoints/hearts_rl_agent.pt'
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    if 'network' in checkpoint:
                        state_dict = checkpoint['network']
                    else:
                        state_dict = checkpoint
                    hidden_size = state_dict.get('input_proj.weight', torch.zeros(512, 1)).shape[0]
                    agent = PPOAgent(hidden_size=hidden_size)
                    agent.load(path)
                    return agent
                except:
                    pass
        return None
    
    # Initialize
    if 'hearts_game' not in st.session_state:
        st.session_state.hearts_game = HeartsGame()
        st.session_state.hearts_game.reset()
        st.session_state.hearts_agent = load_hearts_agent()
        st.session_state.hearts_bots = [SmartBot(), SmartBot(), RandomBot()]
        st.session_state.hearts_message = "Game started! You are South."
        st.session_state.hearts_last_trick = None
        st.session_state.hearts_show_trick = False
        st.session_state.hearts_recorder = []  # Store (obs, action, mask) tuples
    
    game = st.session_state.hearts_game
    agent = st.session_state.hearts_agent
    bots = st.session_state.hearts_bots
    
    if agent is None:
        st.warning("Hearts agent not found. Using SmartBot instead.")
    
    names = ["You (South)", "West", "North", "East"]
    
    # Game over - save human data if won
    if game.game_over:
        winner = game.get_winner()
        if winner == 0:
            st.success("🎉 You Win!")
            # Save winning game data
            if st.session_state.hearts_recorder:
                import pickle
                from datetime import datetime
                os.makedirs('data/human_games', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/human_games/game_{timestamp}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(st.session_state.hearts_recorder, f)
                st.info(f"Saved your winning game to {filepath} for training!")
        else:
            st.error(f"{names[winner]} Wins!")
        
        for i, name in enumerate(names):
            st.write(f"{name}: {game.total_scores[i]} points")
        
        if st.button("🔄 New Game"):
            for key in list(st.session_state.keys()):
                if key.startswith('hearts_'):
                    del st.session_state[key]
            st.rerun()
        return
    
    # Layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 📊 Scores")
        for i, name in enumerate(names):
            current = "👉 " if game.current_player == i else ""
            st.write(f"{current}{name}: {game.total_scores[i]} (round: {game.round_scores[i]})")
    
    with col2:
        st.markdown("### 🎴 Current Trick")
        if game.current_trick:
            for p, c in game.current_trick:
                color = "#ff4757" if c.suit in [Suit.HEARTS, Suit.DIAMONDS] else "#00ff88"
                st.markdown(f"**{names[p]}**: <span style='color:{color};font-size:20px;font-weight:bold'>{RANK_NAMES[c.rank]}{SUIT_NAMES[c.suit]}</span>", unsafe_allow_html=True)
        else:
            st.write("Waiting for first card...")
    
    with col3:
        st.markdown("### 📜 Last Trick")
        if st.session_state.hearts_last_trick:
            trick_cards, winner_idx, points = st.session_state.hearts_last_trick
            cards_html = ""
            for p, c in trick_cards:
                color = "#ff4757" if c.suit in [Suit.HEARTS, Suit.DIAMONDS] else "#00ff88"
                player_type = ["You", "Smart", "RL", "Random"][p]
                cards_html += f"<div style='margin:5px 0'><b>{player_type}:</b> <span style='color:{color};font-size:18px'>{RANK_NAMES[c.rank]}{SUIT_NAMES[c.suit]}</span></div>"
            winner_type = ["You", "Smart", "RL", "Random"][winner_idx]
            st.markdown(f"""
            <div class='last-trick'>
                {cards_html}
                <hr style='border-color:#444;margin:10px 0'>
                <span class='trick-winner'>🏆 {winner_type} won (+{points} pts)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("No tricks played yet")

    
    st.info(st.session_state.hearts_message)
    
    # Your turn
    if game.current_player == 0:
        st.markdown("### 🃏 Your Hand")
        legal = game.get_legal_moves(0)
        hand = sorted(game.players[0].hand, key=lambda c: (c.suit, c.rank))
        
        cols = st.columns(min(len(hand), 13))
        for i, card in enumerate(hand):
            with cols[i % len(cols)]:
                label = f"{RANK_NAMES[card.rank]}{SUIT_NAMES[card.suit]}"
                is_legal = card in legal
                
                if st.button(label, key=f"hearts_card_{i}", disabled=not is_legal, use_container_width=True):
                    # Record move for training data
                    obs = get_obs(game, 0)
                    mask = get_mask(game, 0)
                    st.session_state.hearts_recorder.append((obs, card.to_index(), mask))
                    
                    # Store trick before playing
                    trick_before = list(game.current_trick)
                    
                    winner = game.play_card(0, card)
                    st.session_state.hearts_message = f"You played {label}"

                    
                    if winner is not None:
                        # Calculate points
                        full_trick = trick_before + [(0, card)]
                        points = sum(c.penalty_points() for _, c in full_trick)
                        st.session_state.hearts_last_trick = (full_trick, winner, points)
                        st.session_state.hearts_message = f"🏆 {names[winner]} won the trick! (+{points} pts)"
                    
                    if game.trick_number == 13 and not game.game_over:
                        game.new_round()
                    
                    st.rerun()
    else:
        # Bot turn
        st.markdown("### ⏳ Opponents' Turn...")
        
        while game.current_player != 0 and not game.game_over and game.trick_number < 13:
            # Store trick before playing
            trick_before = list(game.current_trick)
            
            player = game.current_player
            legal = game.get_legal_moves(player)
            
            if player == 2 and agent:
                obs = get_obs(game, player)
                mask = get_mask(game, player)
                action, _, _ = agent.select_action(obs, mask)
                card = Card.from_index(action)
                if card not in legal:
                    card = legal[0]
            else:
                bot = bots[player - 1] if player > 0 else bots[0]
                state = game.get_state(player)
                card = bot.select_card(legal, state)
            
            winner = game.play_card(player, card)
            st.session_state.hearts_message = f"{names[player]} played {RANK_NAMES[card.rank]}{SUIT_NAMES[card.suit]}"
            
            if winner is not None:
                # Show completed trick with delay
                full_trick = trick_before + [(player, card)]
                points = sum(c.penalty_points() for _, c in full_trick)
                st.session_state.hearts_last_trick = (full_trick, winner, points)
                st.session_state.hearts_message = f"🏆 {names[winner]} won the trick! (+{points} pts)"
                
                if game.trick_number == 13 and not game.game_over:
                    game.new_round()
                
                # No delay - just break and rerun
                break
        
        st.rerun()
    
    if st.button("🔄 Reset Game"):
        for key in list(st.session_state.keys()):
            if key.startswith('hearts_'):
                del st.session_state[key]
        st.rerun()


# ============== MAIN ==============
def main():
    st.sidebar.markdown("# 🎮 Game Hub")
    
    game_choice = st.sidebar.radio(
        "Choose a Game:",
        ["🏠 Home", "⭕ Tic-Tac-Toe", "🔴 Connect Four", "⚫ Othello", "♥️ Hearts"],
        index=0
    )
    
    if game_choice == "🏠 Home":
        st.markdown('<div class="game-title">🎮 Game Hub 🎮</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to Game Hub!
        
        Choose a game from the sidebar to play against AI opponents.
        
        | Game | Description | AI Type |
        |------|-------------|---------|
        | ⭕ **Tic-Tac-Toe** | Classic 3x3 grid game | AlphaZero MCTS |
        | 🔴 **Connect Four** | Drop discs to connect 4 | AlphaZero MCTS |
        | ⚫ **Othello** | Flip opponent discs | AlphaZero MCTS |
        | ♥️ **Hearts** | Card game, lowest score wins | PPO RL Agent |
        
        ### Training Status
        """)
        
        games = [
            ("tictactoe", "models/tictactoe_az.pth"),
            ("connect_four", "models/connect_four_az.pth"),
            ("othello", "models/othello_az.pth"),
        ]
        
        for name, path in games:
            status = "✅ Ready" if os.path.exists(path) else "❌ Not trained"
            st.write(f"- **{name}**: {status}")
        
        hearts_paths = ['models/hearts_rl_agent.pt', 'checkpoints/hearts_rl_agent.pt']
        hearts_ready = any(os.path.exists(p) for p in hearts_paths)
        st.write(f"- **Hearts**: {'✅ Ready' if hearts_ready else '❌ Not trained'}")

        
        st.markdown("""
        ### How to Train
        ```bash
        python -m src.main  # Then select game number
        ```
        """)
        
    elif game_choice == "⭕ Tic-Tac-Toe":
        play_tictactoe()
    elif game_choice == "🔴 Connect Four":
        play_connect_four()
    elif game_choice == "⚫ Othello":
        play_othello()
    elif game_choice == "♥️ Hearts":
        play_hearts()


if __name__ == "__main__":
    main()
