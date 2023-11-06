import torch
import time
import torch.nn.functional as F
from sklearn import  metrics
import numpy as np
from tqdm import tqdm, trange
import os
from mydatasets import SQuAD_V2_Dataset
import timeit
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate
from transformers.data.processors.squad import SquadResult,SquadFeatures,SquadExample
from train_eval.evaluate_official2 import eval_squad
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup



def to_list(tensor):
    return tensor.detach().cpu().tolist() if not tensor==torch.Size([0]) else None

def train_and_eval(config,model,train_loader,tokenizer, rank, logger, common_config):
    global_step = 0
    best_f1 = -1
    tr_loss, logging_loss = 0.0, 0.0
    # t_total = len(train_loader) // config.gradient_accumulation_steps * config.nums_epochs
    t_total = config.max_steps * config.batch_size // config.gradient_accumulation_steps
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    logger.info(f"learning rate = {config.learning_rate}")
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

    # tb_writer = SummaryWriter()

    # Training
    logger.info("Local Training Start ...")
    model.zero_grad()
    model.train()
    for step in range(config.max_steps):
        batch = next(train_loader)
        global_step += 1

        batch = tuple(t.to(config.device) for t in batch)

        inputs = {
            'input_ids':       batch[0],
            'attention_mask':  batch[1],
            'token_type_ids' : batch[2],
            'start_positions': batch[3],
            'end_positions':   batch[4]
        }

        span_loss,start_logits,end_logits,type_prob = model(inputs)

        loss = span_loss
        if config.logging_steps > 0 and global_step % config.logging_steps == 0:
            # logger.info('training lr = {}'.format(scheduler.get_lr()[0]))
            logger.info('=========================== training loss {}'.format((tr_loss - logging_loss)/config.logging_steps))
            logging_loss = tr_loss
        
        tr_loss += loss.item()
        loss.backward()
        
        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            # scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            # tqdm.write(str(loss))
    
    logger.info("Local Training finished.")
    
    # Evaluation
    if  (common_config.tag == 1) or ((common_config.tag) % 5 == 0) or (common_config.tag + 1 == common_config.epoch+1):
        logger.info("Evaluation on test set...")
        results = evaluate(config, model, tokenizer, rank)
        for key, value in results.items():
            logger.info('{}:{}'.format(key,value))
        # tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
    # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/config.logging_steps, global_step)
    # logging_loss = tr_loss
    # f1 = results['f1']
    # logger.critical(f"lr = {scheduler.get_lr()[0]}")
    # logger.critical(f"loss = {scheduler.get_lr()[0]}")








def test(config,model,test_iter):
    pass


def evaluate(config, model, tokenizer, rank, prefix=""):
    train_Dataset = SQuAD_V2_Dataset(tokenizer=tokenizer,data_dir=config.dev_path,filename=config.dev_file,is_training=False,config=config,cached_features_file=os.path.join(config.dev_path,"cache_" + config.dev_file.replace("json","data")))
    dataset,examples,features = train_Dataset.dataset,train_Dataset.examples,train_Dataset.features
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


    # Eval!

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1]
            }

            example_indices = batch[3]


            outputs = model(inputs)
        output = [to_list(k) for k in outputs]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)


            start_logits, end_logits = output[1][i],output[2][i]
            result = SquadResult(
                unique_id, start_logits, end_logits
            )

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time

    # Compute predictions
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    output_prediction_file = os.path.join(config.output_dir, "__{}__predictions_{}.json".format(rank, prefix))
    output_nbest_file = os.path.join(config.output_dir, "__{}__nbest_predictions_{}.json".format(rank, prefix))

    output_null_log_odds_file = os.path.join(config.output_dir, "__{}__null_odds_{}.json".format(rank, prefix))

    predictions = compute_predictions_logits(examples, features, all_results, config.n_best_size,
                    config.max_answer_length, config.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, config.verbose_logging,
                    config.version_2_with_negative, config.null_score_diff_threshold,tokenizer)

    # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    #SQuAD 2.0
    results = eval_squad(os.path.join(config.dev_path, config.dev_file), output_prediction_file, output_null_log_odds_file,
                            config.null_score_diff_threshold)
    return results