import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Importiamo la classe wrapper presente nei tuoi file
from simglucose.envs.simglucose_gym_env import T1DSimGymnaisumEnv

def train():
    # 1. Configurazione dell'Ambiente
    # Usiamo il wrapper Gymnasium fornito nei file. 
    # Specifichiamo il paziente e lo scenario personalizzato se necessario.
    # Nota: reward_fun=None usa la reward di default (differenza di Risk Index)
    env = T1DSimGymnaisumEnv(
        patient_name='adolescent#003',
        reward_fun=None, 
        seed=42
    )
    

    # 2. Definizione del Modello PPO
    # MlpPolicy indica che usiamo una rete neurale standard (Multi Layer Perceptron).
    # verbose=1 ci permette di vedere i log durante il training.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01, # Incoraggia l'esplorazione
        tensorboard_log="./ppo_t1d_tensorboard/"
    )

    # 3. Addestramento
    # 1.000.000 di step sono un buon punto di partenza per vedere risultati stabili.
    print("Inizio addestramento PPO...")
    model.learn(total_timesteps=1000000)

    # 4. Salvataggio del modello
    model.save("ppo_adolescent003_basal")
    print("Addestramento completato e modello salvato.")

if __name__ == "__main__":
    train()