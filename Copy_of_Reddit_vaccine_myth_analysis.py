{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of Reddit vaccine myth analysis.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyPu9T3E3QkBchar5u3DCXuR"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "u2NCASrqgvOQ"
      },
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import re\n",
        "import nltk\n",
        "\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "wnl = WordNetLemmatizer()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gkv2Pf6QsVoi",
        "outputId": "98cc8285-9612-4b3e-94dd-fd8cd79bd50c"
      },
      "source": [
        "nltk.download('stopwords')\n",
        "\n",
        "from nltk.corpus import stopwords\n",
        "stop_words = set(stopwords.words(\"english\"))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xKgNb91_g8nU"
      },
      "source": [
        "rvm_df = pd.read_csv('/content/reddit_vm.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QFqlWeRdj6bL"
      },
      "source": [
        "#convert body text to string to remove NaNs and make 1 dtype\n",
        "\n",
        "rvm_df['body'] = rvm_df['body'].astype(str)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aO3LkiIGeg7K"
      },
      "source": [
        "# collect all strings in body text into one string\n",
        "\n",
        "body_all = ''\n",
        "for text in rvm_df['body']:\n",
        "  if text != 'nan':\n",
        "      body_all += ' ' + text\n",
        "\n",
        "# collect all strings in title text into one string\n",
        "\n",
        "title_all = ''\n",
        "for text in rvm_df['title']:\n",
        "  if text != 'nan' and text != 'Comment':\n",
        "      title_all += ' ' + text\n",
        "\n",
        "#remove punctuation using regex\n",
        "\n",
        "body_all = re.sub(r'[^\\w\\s]', '', body_all)\n",
        "title_all = re.sub(r'[^\\w\\s]', '', title_all)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ffCi2GeLfmzB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "7d693c40-cd54-4aca-dec0-0abcb04402dc"
      },
      "source": [
        "# tokenize all body text and convert to NLTK text type\n",
        "\n",
        "nltk.download('punkt')\n",
        "body_text = nltk.Text(nltk.word_tokenize(body_all))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X4d3kIXulOoD"
      },
      "source": [
        "#tokenize body and title text for analysis\n",
        "\n",
        "rvm_df['tokenized_body'] = rvm_df['body'].apply(nltk.word_tokenize)\n",
        "\n",
        "rvm_df['tokenized_title'] = rvm_df['title'].apply(nltk.word_tokenize)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KvLIz3f3TFnc"
      },
      "source": [
        "def toke_lemmatize(list):\n",
        "  \"\"\"\n",
        "  lemmatize a list of tokenized text\n",
        "  \"\"\"\n",
        "  list = [wnl.lemmatize(word) for word in list]\n",
        "  return list"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EkMl_NBVjGf0"
      },
      "source": [
        "def toke_stopwords(list):\n",
        "  \"\"\"\n",
        "  remove stopwords from a list of tokenized text\n",
        "  \"\"\"\n",
        "  list = [word for word in list if not word in stop_words]\n",
        "  return list"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "O_Ufnnyj9Ytu",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d5a1da45-4d9a-438a-e6a4-c5d184b4af0d"
      },
      "source": [
        "nltk.download('wordnet')\n",
        "\n",
        "# lemmatize and remove stopwords from tokenized body and title entries\n",
        "\n",
        "rvm_df['lem_body'] = rvm_df['tokenized_body'].apply(toke_lemmatize)\n",
        "rvm_df['lem_title'] = rvm_df['tokenized_title'].apply(toke_lemmatize)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Package wordnet is already up-to-date!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YaXagc2wjsdg"
      },
      "source": [
        "rvm_df['processed_body'] = rvm_df['lem_body'].apply(toke_stopwords)\n",
        "rvm_df['processed_title'] = rvm_df['lem_title'].apply(toke_stopwords)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QFo8gEjH87S6"
      },
      "source": [
        "# drop columns that have been processed\n",
        "rvm_df = rvm_df.drop(['tokenized_body', 'tokenized_title', 'lem_body', 'lem_title'], axis=1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JLIWUiPw7--S"
      },
      "source": [
        "comments_cleaned = []\n",
        "for text in rvm_df['lem_text']:\n",
        "  for word in text:\n",
        "    if word.isalpha() and word != 'nan':\n",
        "      comments_cleaned.append(word)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QWEt3mYIp6bO"
      },
      "source": [
        "fd_comments = nltk.FreqDist([w.lower() for w in comments_cleaned])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3ItavCxisKoR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5c614bfb-ef2e-4f77-d5bb-be53569871c8"
      },
      "source": [
        "nltk.download('vader_lexicon')\n",
        "\n",
        "from nltk.sentiment import SentimentIntensityAnalyzer\n",
        "sia = SentimentIntensityAnalyzer()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package vader_lexicon to /root/nltk_data...\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vxWFuoymtSi_"
      },
      "source": [
        "rvm_df['sentiment dict'] = rvm_df['body'].apply(sia.polarity_scores)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nkxydBPk_1aF"
      },
      "source": [
        "def sentmax(dict):\n",
        "  maximum = max(dict, key=dict.get)\n",
        "  return maximum"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2FqP3a8AAnUV"
      },
      "source": [
        "rvm_df['sentmax'] = rvm_df['sentiment dict'].apply(sentmax)\n",
        "rvm_df['sentmax'] = rvm_df['sentmax'].astype('str')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 391
        },
        "id": "2PJgzoC2A0PL",
        "outputId": "c59354e9-41cc-41ca-e8e1-1633a15bd4d8"
      },
      "source": [
        "rvm_df.tail()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>title</th>\n",
              "      <th>score</th>\n",
              "      <th>id</th>\n",
              "      <th>url</th>\n",
              "      <th>comms_num</th>\n",
              "      <th>created</th>\n",
              "      <th>body</th>\n",
              "      <th>timestamp</th>\n",
              "      <th>tokenized_text</th>\n",
              "      <th>stopped_text</th>\n",
              "      <th>lem_text</th>\n",
              "      <th>sentiment dict</th>\n",
              "      <th>sentmax</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1509</th>\n",
              "      <td>Comment</td>\n",
              "      <td>1</td>\n",
              "      <td>ek9w4kt</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1.554616e+09</td>\n",
              "      <td>You didn't answer my question.  You said it ca...</td>\n",
              "      <td>2019-04-07 08:54:20</td>\n",
              "      <td>[You, did, n't, answer, my, question, ., You, ...</td>\n",
              "      <td>[You, did, n't, answer, my, question, ., You, ...</td>\n",
              "      <td>[You, n't, answer, question, ., You, said, ca,...</td>\n",
              "      <td>{'neg': 0.0, 'neu': 0.768, 'pos': 0.232, 'comp...</td>\n",
              "      <td>neu</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1510</th>\n",
              "      <td>Comment</td>\n",
              "      <td>2</td>\n",
              "      <td>ek9vhds</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1.554616e+09</td>\n",
              "      <td>Yeah, a long time ago, vaccines weren’t as saf...</td>\n",
              "      <td>2019-04-07 08:45:00</td>\n",
              "      <td>[Yeah, ,, a, long, time, ago, ,, vaccines, wer...</td>\n",
              "      <td>[Yeah, ,, a, long, time, ago, ,, vaccine, were...</td>\n",
              "      <td>[Yeah, ,, long, time, ago, ,, vaccine, ’, safe...</td>\n",
              "      <td>{'neg': 0.122, 'neu': 0.687, 'pos': 0.192, 'co...</td>\n",
              "      <td>neu</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1511</th>\n",
              "      <td>Comment</td>\n",
              "      <td>1</td>\n",
              "      <td>ek9v4u8</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1.554616e+09</td>\n",
              "      <td>So no one has ever been hurt by the mercury in...</td>\n",
              "      <td>2019-04-07 08:40:01</td>\n",
              "      <td>[So, no, one, has, ever, been, hurt, by, the, ...</td>\n",
              "      <td>[So, no, one, ha, ever, been, hurt, by, the, m...</td>\n",
              "      <td>[So, one, ha, ever, hurt, mercury, vaccine, ev...</td>\n",
              "      <td>{'neg': 0.329, 'neu': 0.671, 'pos': 0.0, 'comp...</td>\n",
              "      <td>neu</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1512</th>\n",
              "      <td>Comment</td>\n",
              "      <td>2</td>\n",
              "      <td>ek9ocan</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1.554610e+09</td>\n",
              "      <td>But you do have a point there. Going through t...</td>\n",
              "      <td>2019-04-07 07:02:15</td>\n",
              "      <td>[But, you, do, have, a, point, there, ., Going...</td>\n",
              "      <td>[But, you, do, have, a, point, there, ., Going...</td>\n",
              "      <td>[But, point, ., Going, blood, harder, expel, c...</td>\n",
              "      <td>{'neg': 0.155, 'neu': 0.845, 'pos': 0.0, 'comp...</td>\n",
              "      <td>neu</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1513</th>\n",
              "      <td>Comment</td>\n",
              "      <td>2</td>\n",
              "      <td>ek9oa6b</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>1.554610e+09</td>\n",
              "      <td>Going into the blood can cause non-lethal, but...</td>\n",
              "      <td>2019-04-07 07:01:24</td>\n",
              "      <td>[Going, into, the, blood, can, cause, non-leth...</td>\n",
              "      <td>[Going, into, the, blood, can, cause, non-leth...</td>\n",
              "      <td>[Going, blood, cause, non-lethal, ,, serious, ...</td>\n",
              "      <td>{'neg': 0.202, 'neu': 0.715, 'pos': 0.083, 'co...</td>\n",
              "      <td>neu</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "        title  score  ...                                     sentiment dict sentmax\n",
              "1509  Comment      1  ...  {'neg': 0.0, 'neu': 0.768, 'pos': 0.232, 'comp...     neu\n",
              "1510  Comment      2  ...  {'neg': 0.122, 'neu': 0.687, 'pos': 0.192, 'co...     neu\n",
              "1511  Comment      1  ...  {'neg': 0.329, 'neu': 0.671, 'pos': 0.0, 'comp...     neu\n",
              "1512  Comment      2  ...  {'neg': 0.155, 'neu': 0.845, 'pos': 0.0, 'comp...     neu\n",
              "1513  Comment      2  ...  {'neg': 0.202, 'neu': 0.715, 'pos': 0.083, 'co...     neu\n",
              "\n",
              "[5 rows x 13 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "O6EZzVo6A3XJ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 164
        },
        "outputId": "ab28bf1b-5231-4c25-a49f-ec1ef6f8972d"
      },
      "source": [
        "rvm_df['sentmax'].count(level=str)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-57-b2cca90a9cce>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mrvm_df\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'sentmax'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcount\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mstring\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'string' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uQN0RbZ-UFDo"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}