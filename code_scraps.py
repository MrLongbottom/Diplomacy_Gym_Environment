# not sure about these values, has to do with batching, experience replay and not updating the network after every episode / step
'''
batch_size = 32
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 100
# How often to update the target network
update_target_network = 100
'''

# Earlier scrap based on Q-learning
'''
# Update every fourth frame and once batch size is over 32
if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
    # Get indices of samples for replay buffers
    indices = np.random.choice(range(len(done_history)), size=batch_size)

    # Using list comprehension to sample from replay buffer
    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = [rewards_history[i] for i in indices]
    action_sample = [action_history[i] for i in indices]
    done_sample = tf.convert_to_tensor(
        [float(done_history[i]) for i in indices]
    )

    # Build the updated Q-values for the sampled future states
    # Use the target model for stability
    future_rewards = model_target.predict(state_next_sample)
    # Q value = reward + discount factor * expected future reward
    updated_q_values = rewards_sample + gamma * tf.reduce_max(
        future_rewards, axis=1
    )

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    # Create a mask so we only calculate loss on the updated Q-values
    # updated mask to just be probabilities
    masks = tf.convert_to_tensor(action_sample, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-values
        q_values = model(state_sample)

        # Apply the masks to the Q-values to get the Q-value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        # Calculate loss between new Q-value and old Q-value
        loss = loss_function(updated_q_values, q_action)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

if frame_count % update_target_network == 0:
    # update the the target network with new weights
    model_target.set_weights(model.get_weights())
    # Log details
    template = "running reward: {:.2f} at episode {}, frame count {}"
    print(template.format(running_reward, episode_count, frame_count))

# Limit the state and reward history
if len(rewards_history) > max_memory_length:
    del rewards_history[:1]
    del state_history[:1]
    del state_next_history[:1]
    del action_history[:1]
    del done_history[:1]

if done:
    break

# Update running reward to check condition for solving
episode_reward_history.append(episode_reward)
if len(episode_reward_history) > 100:
    del episode_reward_history[:1]
running_reward = np.mean(episode_reward_history)

episode_count += 1
print("episode: {}, total reward: {:.2f}".format(episode_count, episode_reward))

if episode_count >= 1000:
    model.save('model/test_model')
    break
'''