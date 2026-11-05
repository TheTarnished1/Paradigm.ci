export const CREDIT = "Made by Pascal";

export function PrivacyNotice() {
  return (
    <>
      <p>
        <strong>What Paradigm saves:</strong> your username, a scrambled copy of your password, your assistant setup,
        your chats, the text of documents you upload, and usage counts for rate limits.
      </p>
      <p>
        <strong>Who else handles your data:</strong> Groq receives your messages and document excerpts to write replies.
        A web search service receives search terms when the assistant searches the web. Neon stores your data, and
        Vercel runs the app.
      </p>
      <p>
        <strong>Your choices:</strong> in Settings, under Account, you can download your data or delete your account.
        Deleting removes your account, chats, setup and documents.
      </p>
      <p>Don&apos;t enter passwords, payment details or other highly sensitive information in chats.</p>
    </>
  );
}
