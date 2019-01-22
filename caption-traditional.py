import torch
import torch.nn.functional as F
import json
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gr
import skimage.transform
from scipy.misc import imread, imresize
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MainWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="")
        self.set_border_width(30)
        layout = Gtk.Box(spacing = 6)
        self.add(layout)

        button = Gtk.Button("Choose File")
        button.connect("clicked", self.on_file_clicked)
        layout.add(button)

    def on_file_clicked(self, widget):

        dialog = Gtk.FileChooserDialog("select a file", self, Gtk.FileChooserAction.OPEN,
                                       ("cancle", Gtk.ResponseType.CANCEL,
                                        "ok", Gtk.ResponseType.OK))

        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            print("OK")
            print(dialog.get_filename())
            the_image_path = dialog.get_filename()
            img = the_image_path
            model = './BEST_tra_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
            checkpoint = torch.load(model, map_location='cpu')
            decoder = checkpoint['decoder']
            decoder = decoder.to(device)
            decoder.eval()
            encoder = checkpoint['encoder']
            encoder = encoder.to(device)
            encoder.eval()

            word_map = './data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
            with open(word_map, 'r') as j:
                word_map = json.load(j)
            rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

            # Encode, decode with attention and beam search
            seq = caption_image_beam_search(encoder, decoder, img, word_map, beam_size = 5)

            # Visualize caption and attention of best sequence
            visualize_att(img, seq, rev_word_map, smooth = True)

        else:
            print("Cancel")

        dialog.destroy()


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=5):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, encoder_dim)
    encoder_dim = encoder_out.size(1)
    encoder_out = encoder_out.expand(k, encoder_dim)

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    seqs = k_prev_words  # (k, 1)
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    h = torch.zeros([k, 512])
    c = torch.zeros([k, 512])
    h, c = decoder.decode_step(encoder_out, (h, c))
    step = 1

    while True:

        embeddings_cap = decoder.embedding(k_prev_words).squeeze(1) # (s, embed_dim)

        h, c = decoder.decode_step(embeddings_cap, (h, c))

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    print(seq)
    return seq


def visualize_att(image_path, seq,  rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)

    words = [rev_word_map[ind] for ind in seq]
    print(words)

    new_words = words[1:len(words) - 1]
    print(new_words)
    new_words = new_words
    sentence = ' '.join(new_words[0: len(new_words)])
    print(sentence)
    plt.imshow(np.asarray(image))
    plt.text(15, -15, sentence)
    plt.show()


window = MainWindow()
window.connect("delete-event", Gtk.main_quit)
window.show_all()
Gtk.main()
