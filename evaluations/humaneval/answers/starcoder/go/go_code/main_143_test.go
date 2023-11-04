package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestWordsInSentence(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("is", WordsInSentence("This is a test"))
    assert.Equal("go for", WordsInSentence("lets go for swimming"))
    assert.Equal("there is no place", WordsInSentence("there is no place available here"))
    assert.Equal("Hi am Hussein", WordsInSentence("Hi I am Hussein"))
    assert.Equal("go for it", WordsInSentence("go for it"))
    assert.Equal("", WordsInSentence("here"))
    assert.Equal("is", WordsInSentence("here is"))
}
