import logging

from telegram import ReplyKeyboardRemove
from telegram.ext import (Updater, CommandHandler, Filters)

from tg_credentials import credentials

from ranepa_flask_wrapper.flask_wrapper import flask_wrapper

# credentials = {"token": "your-token",
#                "user_id": 100000000}


class TelegramCaptchaBot(object):
    """  A class for sending captcha images and getting replies
    Supports the following commands:
     /start: activate and get a reply with all command options
     /help: get a reply with all command options
     /captcha: get a reply with the captcha text
    # Arguments
        token: String, a telegram bot token
        user_id: Integer. Specifying a telegram user id
                 will filter all incoming
                 commands to allow access only to a specific user.
                 Optional, though highly recommended.
    """

    def __init__(self, token, user_id=None):
        assert isinstance(token, str), 'Token must be of type string'
        assert user_id is None or isinstance(
            user_id, int), 'user_id must be of type int (or None)'

        self.token = token  # bot token
        self.user_id = user_id  # id of the user with access

        self.filters = None
        self.chat_id = None  # chat id, will be fetched during /start command

        # placeholder status message
        self._status_message = "No status message was set"

        self.updater = None

        self.captchas = []

        # Enable logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Message to display on /start and /help commands
        self.startup_message = """
        Hi, I'm the Captcha bot! I will send you new captchas to recognize.
        send /start to activate captcha sending
        send /help to see all options.
        Send /status to get the latest results.
        Send /captcha to recognize captcha text
        Send /send_captcha to send new captcha for recognition
        """
        self.activate_bot()

    def activate_bot(self):
        """ Function to initiate the Telegram bot """
        self.updater = Updater(self.token)  # setup updater
        dp = self.updater.dispatcher  # Get the dispatcher to register handlers
        dp.add_error_handler(self.error)  # log all errors

        self.filters = Filters.user(
            user_id=self.user_id) if self.user_id else None
        # Command and conversation handles
        dp.add_handler(
            CommandHandler("start", self.start, filters=self.filters))
        dp.add_handler(
            CommandHandler("help", self.help, filters=self.filters))
        dp.add_handler(
            CommandHandler("status", self.status, filters=self.filters))
        dp.add_handler(
            CommandHandler("captcha", self.captcha, filters=self.filters,
                           pass_args=True,
                           pass_job_queue=True,
                           pass_chat_data=True))
        dp.add_handler(
            CommandHandler(
                "send_picture", self.send_picture, filters=self.filters))

        # Start the Bot
        self.updater.start_polling()
        self.bot_active = True

        # Uncomment next line while debugging
        # updater.idle()

    def stop_bot(self):
        """ Function to stop the bot """
        self.updater.stop()
        self.bot_active = False

    def start(self, bot, update):
        """ Telegram bot callback for the /start command.
        Fetches chat_id and sends startup message"""
        update.message.reply_text(
            self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id
        self.verbose = True

    def help(self, bot, update):
        """ Telegram bot callback for the /help command.
        Replies the startup message"""
        update.message.reply_text(
            self.startup_message, reply_markup=ReplyKeyboardRemove())
        self.chat_id = update.message.chat_id

    def error(self, bot, update, error):
        """Log Errors caused by Updates."""
        self.logger.warning('Update "%s" caused error "%s"', update, error)

    def send_message(self, txt):
        """ Function to send a Telegram message to user
         # Arguments
            txt: String, the message to be sent
        """
        assert isinstance(txt, str), 'Message text must be of type string'
        if self.chat_id is not None:
            self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
        else:
            print('Send message failed, user did not send /start')

    def get_captchas(self, *args, **kwargs):
        return self.captchas

    # Plot loss history
    def send_picture(self, pic, txt=None, binary=True, *args, **kwargs):
        """ Telegram bot callback for the /plot command.
            Sends image with text to the chat"""
        # Sent image to user
        print(pic)
        if not binary:
            pic = open(pic, 'rb')
        if pic:
            if txt:
                self.updater.bot.send_message(chat_id=self.chat_id, text=txt)
            self.updater.bot.send_photo(chat_id=self.chat_id,
                                        photo=pic)
        return self.captchas

    def set_status(self, txt):
        """ Function to set a status message to be returned
            by the /status command """
        assert isinstance(txt, str), 'Status Message must be of type string'
        self._status_message = txt

    def status(self, bot, update):
        """ Telegram bot callback for the /status command.
        Replies with the latest status"""
        update.message.reply_text(self._status_message)

    def captcha(self, bot, update, **kwargs):
        """ Telegram bot callback for the /captcha command.
            Appends text to the status"""
        # captcha = update.message.from_user
        # update.message.reply_text("Please enter captcha and the worker id")
        # self.captchas.append(captcha)
        # text_caps = ' '.join(context.args).upper()
        if kwargs and kwargs["args"]:
            captcha = " ".join(kwargs["args"])
            self.updater.bot.send_message(chat_id=self.chat_id, text=captcha)
            self.captchas.append(captcha)
        else:
            self.updater.bot.send_message(
                chat_id=self.chat_id, text="Please provide the captcha text")


if __name__ == "__main__":
    bot = TelegramCaptchaBot(**credentials)
    routing = {"get_captchas": bot.get_captchas,
               "send_picture": bot.send_picture}
    kwargs = {"binary": False}
    bot.activate_bot()
    flask_wrapper(routing=routing, port=5034, kwargs=kwargs)

    # url = "http://127.0.0.1:5034/"

    # params = {"method": "send_picture", "txt": "0", "binary": False}
    # files = [
    #     ('pic',
    #      ("captcha_1.jpg", open("captcha_1.jpg", "rb"),
    #       'application/octet')),
    #     ('datas', ('datas', json.dumps(params), 'application/json')),
    # ]
    # r = requests.post(url, files=files)
