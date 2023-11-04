package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestStrlen(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(0, Strlen(""))
    assert.Equal(1, Strlen("x"))
    assert.Equal(9, Strlen("asdasnakj"))
}
